"""The interactive review application: the step between ranking and filing.

`picklikeme review --input <folder>` serves a local page where the photographer
sees the model's ordering, corrects it, and finally moves the files on disk.

Deliberately outside `picklikeme.analyzer`: the analyzer is a read-only
reporting tool that must never move a file (see `organize.py`'s module
docstring), while review's whole purpose is to end in a move. Review is the one
component that legitimately imports both the analyzer's read-side helpers
(thumbnails, detector boxes, the annotation store) and the ranking side's
`organize`.

Nothing here performs inference. The ranking is finished before review begins.
"""
