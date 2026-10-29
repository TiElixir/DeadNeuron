import numpy as np
import pytest

from neuronlab.recording import record_set
from neuronlab.storage import Checkpoint, RunStore, settings_hash


def test_arrays_and_json_round_trip(tmp_path):
    st = RunStore(tmp_path, "fp", "ph", {"k": 1})
    a = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.array([True, False])}
    st.save_arrays("scores", a, {"n": 3, "note": "hi"})
    arrays, meta = st.load_arrays("scores")
    assert np.array_equal(arrays["x"], a["x"]) and arrays["b"].dtype == bool and meta == {"n": 3, "note": "hi"}
    st.save_json("m", {"a": [1, 2]})
    assert st.load_json("m") == {"a": [1, 2]} and st.has("m") and st.load_arrays("nope") is None


def test_key_depends_on_model_preset_and_settings(tmp_path):
    a = RunStore(tmp_path, "fp", "ph", {"k": 1}).dir
    assert a == RunStore(tmp_path, "fp", "ph", {"k": 1}).dir
    assert len({a, RunStore(tmp_path, "fp2", "ph", {"k": 1}).dir, RunStore(tmp_path, "fp", "ph2", {"k": 1}).dir,
                RunStore(tmp_path, "fp", "ph", {"k": 2}).dir}) == 4
    assert settings_hash({"a": 1, "b": 2}) == settings_hash({"b": 2, "a": 1})


def test_recording_round_trip_stores_float16(lab, preset, tmp_path):
    rec = record_set(lab, preset.positive[:4], batch_size=2)
    st = RunStore(tmp_path, "fp", "ph", {})
    st.save_recording("pos", rec)
    back = st.load_recording("pos")
    assert back.acts.dtype == np.float16 and back.mean.dtype == np.float32
    assert np.array_equal(back.acts, rec.acts) and np.array_equal(back.mean, rec.mean)
    assert np.array_equal(back.resid, rec.resid) and back.texts == rec.texts and back.n == 4
    assert (st.dir / "pos.acts.npy").exists()


def test_failed_write_keeps_previous_file_and_leaves_no_temp(tmp_path, monkeypatch):
    st = RunStore(tmp_path, "fp", "ph", {})
    st.save_arrays("scores", {"x": np.ones(3)}, {"v": 1})

    def broken_savez(f, **kw):
        f.write(b"partial garbage")
        raise OSError("disk full")

    monkeypatch.setattr("neuronlab.storage.np.savez", broken_savez)
    with pytest.raises(OSError):
        st.save_arrays("scores", {"x": np.zeros(3)}, {"v": 2})
    monkeypatch.undo()
    arrays, meta = st.load_arrays("scores")
    assert arrays["x"].tolist() == [1, 1, 1] and meta == {"v": 1}
    assert not list(st.dir.glob("*.tmp"))


def test_checkpoint_resume_after_forced_stop(tmp_path):
    """A job that crashes mid-way resumes from its checkpoint and ends with the same result."""
    st = RunStore(tmp_path, "fp", "ph", {})
    items = list(range(10))

    def job(crash_at=None):
        ck = Checkpoint(st, "sweep")
        got = ck.load()
        done = int(got[1]["done"]) if got else 0
        out = got[0]["out"].copy() if got else np.zeros(10)
        for i in range(done, 10):
            if crash_at is not None and i == crash_at:
                raise KeyboardInterrupt
            out[i] = items[i] ** 2
            if (i + 1) % 3 == 0:
                ck.save({"out": out}, {"done": i + 1})
        return out

    with pytest.raises(KeyboardInterrupt):
        job(crash_at=7)
    assert Checkpoint(st, "sweep").load()[1]["done"] == 6  # last full checkpoint before the crash
    assert np.array_equal(job(), np.array(items, float) ** 2)
