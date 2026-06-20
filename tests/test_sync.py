import numpy as np
import pytest

from golfie_cv.sync import align_audio_data


def test_align_audio_data_recovers_exact_offset():
    rate = 44100
    # Generate 3 seconds of background noise
    np.random.seed(42)
    noise_a = np.random.normal(0, 0.01, rate * 3).astype(np.float32)
    noise_b = np.random.normal(0, 0.01, rate * 3).astype(np.float32)

    # Place a clean transient (gaussian spike) at known positions
    # Event in A is at 1.5 seconds
    # Event in B is at 1.8 seconds
    # So A's event is earlier than B's relative to their video starts
    # offset_sec = t_a - t_b = 1.5 - 1.8 = -0.3 seconds
    idx_a = int(rate * 1.5)
    idx_b = int(rate * 1.8)

    spike = np.zeros(int(rate * 0.2), dtype=np.float32)
    spike[len(spike) // 2] = 0.8

    noise_a[idx_a - len(spike)//2 : idx_a + len(spike)//2] += spike
    noise_b[idx_b - len(spike)//2 : idx_b + len(spike)//2] += spike

    offset_sec, confidence, peak_a, peak_b = align_audio_data(noise_a, rate, noise_b, rate)

    # The recovered offset should be very close to -0.3s
    assert offset_sec == pytest.approx(-0.3, abs=1e-4)
    # The confidence should be very high (close to 1.0) because it's a clean peak
    assert confidence > 0.8
    # The peaks should match the transient indices
    assert peak_a == pytest.approx(idx_a, abs=20)
    assert peak_b == pytest.approx(idx_b, abs=20)


def test_align_audio_data_handles_stereo():
    rate = 16000
    # 2 seconds of audio
    noise = np.random.normal(0, 0.01, (rate * 2, 2)).astype(np.float32)
    
    # Put a spike in channel 0
    idx = int(rate * 1.0)
    noise[idx, 0] = 1.0
    noise[idx, 1] = 0.5  # stereo channel 1 has lower amplitude

    # Since they are the same signal with zero relative offset:
    offset_sec, confidence, peak_a, peak_b = align_audio_data(noise, rate, noise, rate)
    assert offset_sec == pytest.approx(0.0, abs=1e-5)
    assert confidence > 0.8


def test_align_audio_data_handles_int16():
    rate = 16000
    # Create int16 signals
    data_a = np.zeros(rate * 2, dtype=np.int16)
    data_b = np.zeros(rate * 2, dtype=np.int16)

    # Spike at 0.8s in A, 1.2s in B
    # offset_sec = t_a - t_b = 0.8 - 1.2 = -0.4s
    idx_a = int(rate * 0.8)
    idx_b = int(rate * 1.2)

    data_a[idx_a] = 30000
    data_b[idx_b] = 30000

    offset_sec, confidence, peak_a, peak_b = align_audio_data(data_a, rate, data_b, rate)
    assert offset_sec == pytest.approx(-0.4, abs=1e-5)
    assert confidence > 0.8
