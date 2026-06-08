"""
Tests for import_icons' image helpers (no network, no GUI).

Run:  python tests/test_import.py   (or: pytest -q)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import import_icons as ii


def _skip_no_cv():
    if ii.cv2 is None or ii.np is None:
        print("(skipped: opencv not installed)")
        return True
    return False


def test_composite_on_bg_fills_transparent():
    if _skip_no_cv():
        return
    np = ii.np
    img = np.zeros((2, 2, 4), np.uint8)
    img[0, 0] = (200, 100, 50, 255)        # opaque BGRA
    img[1, 1] = (0, 0, 0, 0)               # fully transparent
    out = ii.composite_on_bg(img, (10, 20, 30))    # bg in BGR
    assert tuple(int(v) for v in out[0, 0]) == (200, 100, 50)   # opaque kept
    assert tuple(int(v) for v in out[1, 1]) == (10, 20, 30)     # transparent -> bg
    assert out.shape == (2, 2, 3)          # alpha dropped


def test_composite_passthrough_bgr():
    if _skip_no_cv():
        return
    np = ii.np
    img = np.full((3, 3, 3), 7, np.uint8)
    out = ii.composite_on_bg(img, (0, 0, 0))
    assert out.shape == (3, 3, 3) and int(out[0, 0, 0]) == 7


def test_inscribed_square_is_centered_square():
    if _skip_no_cv():
        return
    np = ii.np
    img = np.zeros((100, 100, 3), np.uint8)
    out = ii.inscribed_square(img)
    h, w = out.shape[:2]
    assert h == w
    assert 60 <= h <= 72                    # ~100/sqrt(2) = 70


def test_square_resize():
    if _skip_no_cv():
        return
    np = ii.np
    out = ii.square_resize(np.zeros((200, 120, 3), np.uint8), 96)
    assert out.shape == (96, 96, 3)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
