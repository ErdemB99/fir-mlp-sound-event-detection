
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr,
    n_fft,
    n_mels,
    fmin=50.0,
    fmax=None
):
    if fmax is None:
        fmax = sr / 2.0

    if not (0 <= fmin < fmax <= sr / 2.0):
        raise ValueError("Invalid frequency range.")

    if n_fft < 2 or n_mels < 1:
        raise ValueError("Invalid FFT or mel configuration.")

    mel_points = np.linspace(
        hz_to_mel(fmin),
        hz_to_mel(fmax),
        n_mels + 2
    )

    hz_points = mel_to_hz(mel_points)
    frequencies = np.fft.rfftfreq(
        n_fft,
        d=1.0 / sr
    )

    filters = np.zeros(
        (n_mels, len(frequencies)),
        dtype=np.float32
    )

    for m in range(n_mels):
        left, center, right = hz_points[m:m + 3]

        rising = (
            (frequencies - left)
            / max(center - left, 1e-12)
        )

        falling = (
            (right - frequencies)
            / max(right - center, 1e-12)
        )

        filters[m] = np.maximum(
            0.0,
            np.minimum(rising, falling)
        )

    sums = filters.sum(axis=1, keepdims=True)
    filters /= np.maximum(sums, 1e-12)

    return filters


def extract_features(
    audio,
    sr,
    n_fft=2048,
    hop_length=960,
    n_mels=32
):
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError(
            f"Expected 4-channel FOA audio, got {audio.shape}"
        )

    if not (0 < hop_length <= n_fft):
        raise ValueError("Invalid hop length.")

    if len(audio) < n_fft:
        raise ValueError("Audio shorter than FFT window.")

    filters = mel_filterbank(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels
    )

    channel_features = []
    frame_times = None

    for channel in range(4):
        _, times, spectrum = stft(
            audio[:, channel],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
            boundary=None,
            padded=False
        )

        power = np.abs(spectrum) ** 2
        mel = filters @ power

        logmel = np.log(mel + 1e-10).T

        channel_features.append(logmel)

        if frame_times is None:
            frame_times = times

    features = np.concatenate(
        channel_features,
        axis=1
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Non-finite features detected.")

    return features, frame_times


def read_events(csv_path):
    events = []

    with Path(csv_path).open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue

            if len(row) < 3:
                raise ValueError(
                    f"Invalid annotation row in {csv_path}: {row}"
                )

            try:
                start = float(row[1])
                end = float(row[2])

            except ValueError:
                # Accept a header on the first row.
                if not events and row[0].strip().lower() in (
                    "label", "class", "event"
                ):
                    continue

                raise ValueError(
                    f"Invalid annotation in {csv_path}: {row}"
                )

            if not (0 <= start <= end):
                raise ValueError(
                    f"Invalid event timestamps: {row}"
                )

            events.append((
                row[0].strip(),
                start,
                end
            ))

    return events


def build_label_map(csv_files):
    classes = set()

    for csv_path in csv_files:
        for label, _, _ in read_events(csv_path):
            classes.add(label)

    return {
        label: index
        for index, label in enumerate(sorted(classes))
    }


def create_frame_labels(
    events,
    frame_times,
    label_map
):
    labels = np.zeros(
        (len(frame_times), len(label_map)),
        dtype=np.float32
    )

    for label, start, end in events:
        if label not in label_map:
            raise ValueError(
                f"Unknown sound event class: {label}"
            )

        class_index = label_map[label]

        active = (
            (frame_times >= start)
            & (frame_times < end)
        )

        labels[active, class_index] = 1.0

    return labels


def main():
    parser = argparse.ArgumentParser(
        description="TAU 2019 FOA audio preprocessing"
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=RAW / "foa_dev"
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=RAW / "metadata_dev"
    )

    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=DATA / "training_par"
    )

    parser.add_argument(
        "--label-dir",
        type=Path,
        default=DATA / "training_target"
    )

    parser.add_argument(
        "--n-mels",
        type=int,
        default=32
    )

    parser.add_argument(
        "--n-fft",
        type=int,
        default=2048
    )

    parser.add_argument(
        "--hop-length",
        type=int,
        default=960
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000
    )

    args = parser.parse_args()

    audio_dir = args.audio_dir
    metadata_dir = args.metadata_dir

    feature_dir = args.feature_dir
    label_dir = args.label_dir

    wav_files = sorted(audio_dir.glob("*.wav"))
    csv_files = sorted(metadata_dir.glob("*.csv"))

    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in {audio_dir}"
        )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {metadata_dir}"
        )

    if args.n_mels * 4 != 128:
        raise ValueError(
            "Current network requires 128 input features."
        )

    label_map = build_label_map(csv_files)

    if len(label_map) != 11:
        raise ValueError(
            f"Expected 11 classes, found {len(label_map)}."
        )

    feature_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Do not silently mix old and new preprocessing.
    if (
        list(feature_dir.glob("*.npy"))
        or list(label_dir.glob("*.npy"))
    ):
        raise RuntimeError(
            "Output folders already contain NPY files. "
            "Use empty output folders for preprocessing."
        )

    # Save the class mapping beside the dataset.
    mapping_path = DATA / "label_map.json"

    if mapping_path.exists():
        with mapping_path.open(
            "r", encoding="utf-8"
        ) as f:
            existing_map = json.load(f)

        if existing_map != label_map:
            raise ValueError(
                "Existing label_map.json is incompatible."
            )

    else:
        with mapping_path.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                label_map,
                f,
                indent=2
            )

    print("\nCLASS MAPPING")
    print(json.dumps(label_map, indent=2))

    converted = 0
    skipped = 0

    for wav_path in wav_files:
        csv_path = metadata_dir / (
            wav_path.stem + ".csv"
        )

        if not csv_path.is_file():
            print(
                f"[SKIP] Missing annotation: {wav_path.name}"
            )
            skipped += 1
            continue

        audio, sr = sf.read(
            wav_path,
            always_2d=True,
            dtype="float32"
        )

        if sr != args.sample_rate:
            raise ValueError(
                f"Unexpected sample rate in {wav_path.name}: "
                f"{sr} instead of {args.sample_rate}"
            )

        X, frame_times = extract_features(
            audio=audio,
            sr=sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels
        )

        events = read_events(csv_path)

        Y = create_frame_labels(
            events=events,
            frame_times=frame_times,
            label_map=label_map
        )

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Frame mismatch: {wav_path.name}"
            )

        np.save(
            feature_dir / (wav_path.stem + ".npy"),
            X
        )

        np.save(
            label_dir / (wav_path.stem + ".npy"),
            Y
        )

        converted += 1

        print(
            f"[OK] {wav_path.name} | "
            f"Features: {X.shape} | "
            f"Labels: {Y.shape}"
        )

    print("\nPREPROCESSING FINISHED")
    print("Converted:", converted)
    print("Skipped:", skipped)
    print("Features:", feature_dir)
    print("Labels:", label_dir)


if __name__ == "__main__":
    main()
