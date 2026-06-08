"""
Tests for fetch_templates' pure helpers (no network).
The actual download path is exercised manually / in setup, not in CI.

Run:  python tests/test_fetch.py   (or: pytest -q)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fetch_templates as ft

FAKE = [
    {"hero_name": "None", "uid": "null", "portrait": ""},        # sentinel
    {"hero_name": "X.Borg", "uid": "xborg", "portrait": "http://x/xborg.png"},
    {"hero_name": "Minsitthar", "uid": "minsitthar", "portrait": "http://x/m.png"},
    {"hero_name": "Yi Sun-shin", "uid": "yi-sun-shin", "portrait": "http://x/y.png"},
]


def test_norm():
    assert ft._norm("X.Borg") == "xborg"
    assert ft._norm("Yi Sun-shin") == "yisunshin"
    assert ft._norm("  Aamon ") == "aamon"


def test_placeholder_filtered_from_index():
    idx = ft.build_index(FAKE, ("hero_name", "uid"))
    assert "null" not in idx and "none" not in idx
    assert "xborg" in idx and "yisunshin" in idx


def test_resolve_exact_and_fuzzy():
    idx = ft.build_index(FAKE, ("hero_name", "uid"))
    assert ft.resolve("X.Borg", idx)["uid"] == "xborg"          # exact
    # our DB spells it "Minsithar"; source has "Minsitthar" -> fuzzy match
    assert ft.resolve("Minsithar", idx)["uid"] == "minsitthar"
    assert ft.resolve("Totally Fake Hero", idx) is None


def test_safe_filename():
    assert ft._safe_filename("X.Borg") == "x.borg.png"
    assert ft._safe_filename("Yi Sun-shin") == "yi sun-shin.png"
    assert "/" not in ft._safe_filename("a/b:c")


def test_dig():
    assert ft._dig({"data": {"heroes": [1, 2]}}, "data.heroes") == [1, 2]
    assert ft._dig([1, 2], None) == [1, 2]


def test_square_centre_crops():
    if ft.cv2 is None or ft.np is None:
        print("(skipped test_square: opencv not installed)")
        return
    img = ft.np.zeros((200, 120, 3), ft.np.uint8)
    out = ft._square(img, 96)
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
