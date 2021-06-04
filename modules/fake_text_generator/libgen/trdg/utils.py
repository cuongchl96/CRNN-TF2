"""
Utility functions
"""

import os


def load_dict(lang):
    """Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(
        os.path.join(os.path.dirname(__file__), "dicts", lang + ".txt"),
        "r",
        encoding="utf8",
        errors="ignore",
    ) as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
    return lang_dict


def load_fonts(lang):
    """Load all fonts in the fonts directories
    """

    if lang == "cn":
        return [
            os.path.join(os.path.dirname(__file__), "fonts/cn", font)
            for font in os.listdir(os.path.join(os.path.dirname(__file__), "fonts/cn"))
        ]
    elif lang == "vi":
        return [
            os.path.join(os.path.dirname(__file__), "fonts/vi", font)
            for font in os.listdir(os.path.join(os.path.dirname(__file__), "fonts/vi"))
        ]
    elif lang == "vi_basic":
        return [
            os.path.join(os.path.dirname(__file__), "fonts/vi_basic", font)
            for font in os.listdir(os.path.join(os.path.dirname(__file__), "fonts/vi_basic"))
        ]
    elif lang == "cmnd":
        return [
            os.path.join(os.path.dirname(__file__), "fonts/cmnd", font)
            for font in os.listdir(os.path.join(os.path.dirname(__file__), "fonts/cmnd"))
        ]
    else:
        return [
            os.path.join(os.path.dirname(__file__), "fonts/latin", font)
            for font in os.listdir(
                os.path.join(os.path.dirname(__file__), "fonts/latin")
            )
        ]
