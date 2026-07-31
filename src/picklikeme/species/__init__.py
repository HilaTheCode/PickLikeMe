"""Arrange by Species: an independent, optional workflow that runs *after*
Review/Arrange has already filed a shoot into its Keep folder.

    RAW Images -> Detector -> Selection Model -> Review -> Arrange -> Keep Folder
                                                                          |
                                                            Keep Folder --+
                                                                          v
                                                        Arrange by Species (this package)
                                                                          v
                                                     Species Classification Engine
                                                                          v
                                                   Species folders, files moved

Nothing in this package is imported by `analyzer`, `review`, `rank`, or
`train`, and nothing here imports from them either (organize.unique_destination
and analyzer.contactsheets.load_source_image are the only things reused, both
already-generic file/decode helpers with no detector or ranking logic in
them). The detector and the selection model are exactly as they were before
this package existed - this is a consumer of their *final output* (the Keep
folder), not a participant in producing it.

    classifier.py   - the pluggable SpeciesClassifier boundary + SpeciesPrediction
    bioclip_classifier.py - the default local, offline classifier (BioCLIP-2)
    cache.py        - species predictions, memoised by content identity so a
                       later move/rename/reclassify never repeats the work
    arrange.py       - the workflow itself: classify, decide a folder, move
    cli.py           - `picklikeme arrange-species --input <keep folder>`
"""
