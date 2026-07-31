"""Loupe (zoom) review dialog - fast per-image Keep/Reject/Neutral triage.

Mirrors the browser Lightbox's core workflow: zoom/pan the current image,
apply a Keep/Reject/Neutral decision with an optional reason, move to the
next image in the filtered set, and optionally save a quick JPEG for
sharing before any RAW development happens.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ...analyzer.annotations import (
    REVIEW_REASON_BAD_QUALITY,
    REVIEW_REASON_CLEAR_EYES_SEEN,
    REVIEW_REASON_EYES_NOT_SEEN,
    REVIEW_REASON_GOOD_QUALITY,
    REVIEW_REASON_OTHER,
)
from ..services import ReviewService

REASON_LABELS: dict[str, str] = {
    REVIEW_REASON_EYES_NOT_SEEN: "Eyes not seen",
    REVIEW_REASON_CLEAR_EYES_SEEN: "Clear eyes seen",
    REVIEW_REASON_GOOD_QUALITY: "Overall good quality",
    REVIEW_REASON_BAD_QUALITY: "Overall bad quality",
    REVIEW_REASON_OTHER: "Other",
}


class _ZoomView(QGraphicsView):
    """A QGraphicsView that zooms with the mouse wheel and pans by dragging."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, path: Path) -> None:
        self._scene.clear()
        pixmap = QPixmap(str(path))
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(pixmap.rect())
        self.resetTransform()
        if not pixmap.isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override signature
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


class LoupeDialog(QDialog):
    """Full-screen-style zoom review for a filtered set of images."""

    def __init__(
        self,
        *,
        service: ReviewService,
        image_paths: list[str],
        start_index: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if not image_paths:
            raise ValueError("image_paths must not be empty")
        self.service = service
        self.image_paths = image_paths
        self.index = max(0, min(start_index, len(image_paths) - 1))
        self.setWindowTitle("Loupe")
        self.resize(1100, 800)

        self._view = _ZoomView(self)

        self._reason_combo = QComboBox(self)
        for value, label in REASON_LABELS.items():
            self._reason_combo.addItem(label, value)
        self._reason_combo.currentIndexChanged.connect(self._on_reason_changed)

        self._reason_note = QLineEdit(self)
        self._reason_note.setPlaceholderText("Note (only used when reason is Other)")
        self._reason_note.setVisible(False)

        self._name_label = QLabel(self)
        self._counter_label = QLabel(self)

        prev_btn = QPushButton("< Prev", self)
        keep_btn = QPushButton("Keep (K)", self)
        reject_btn = QPushButton("Reject (R)", self)
        neutral_btn = QPushButton("Neutral (N)", self)
        save_jpeg_btn = QPushButton("Save as JPEG", self)
        next_btn = QPushButton("Next >", self)
        close_btn = QPushButton("Close", self)

        prev_btn.clicked.connect(self._go_prev)
        keep_btn.clicked.connect(lambda: self._apply_status("keep"))
        reject_btn.clicked.connect(lambda: self._apply_status("reject"))
        neutral_btn.clicked.connect(lambda: self._apply_status("neutral"))
        save_jpeg_btn.clicked.connect(self._save_jpeg)
        next_btn.clicked.connect(lambda: self._go_next())
        close_btn.clicked.connect(self.accept)

        controls = QHBoxLayout()
        controls.addWidget(prev_btn)
        controls.addWidget(keep_btn)
        controls.addWidget(reject_btn)
        controls.addWidget(neutral_btn)
        controls.addWidget(QLabel("Reason:"))
        controls.addWidget(self._reason_combo)
        controls.addWidget(self._reason_note, 1)
        controls.addWidget(save_jpeg_btn)
        controls.addWidget(next_btn)
        controls.addWidget(close_btn)

        top_bar = QHBoxLayout()
        top_bar.addWidget(self._name_label)
        top_bar.addStretch(1)
        top_bar.addWidget(self._counter_label)

        layout = QVBoxLayout(self)
        layout.addLayout(top_bar)
        layout.addWidget(self._view, 1)
        layout.addLayout(controls)

        self._load_current()

    def _on_reason_changed(self) -> None:
        self._reason_note.setVisible(self._reason_combo.currentData() == REVIEW_REASON_OTHER)

    def _current_path(self) -> str:
        return self.image_paths[self.index]

    def _load_current(self) -> None:
        path = self._current_path()
        try:
            preview = self.service.preview_path(path)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not crash the loupe
            QMessageBox.warning(self, "Loupe", f"Could not load preview:\n{exc}")
            return
        self._view.set_image(preview)
        self._counter_label.setText(f"{self.index + 1} / {len(self.image_paths)}")
        self._name_label.setText(Path(path).name)

    def _apply_status(self, status: str) -> None:
        reason = self._reason_combo.currentData() if status != "neutral" else None
        note = self._reason_note.text().strip() or None if reason == REVIEW_REASON_OTHER else None
        try:
            self.service.set_review_status(self._current_path(), status, reason=reason, reason_note=note)
        except Exception as exc:  # noqa: BLE001 - surfaced to the photographer, not fatal
            QMessageBox.warning(self, "Loupe", f"Could not save decision:\n{exc}")
            return
        self._go_next(auto=True)

    def _go_next(self, *, auto: bool = False) -> None:
        if self.index + 1 < len(self.image_paths):
            self.index += 1
            self._load_current()
        elif auto:
            self.accept()

    def _go_prev(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def _save_jpeg(self) -> None:
        path = self._current_path()
        suggested = Path(path).with_suffix(".jpg").name
        destination, _ = QFileDialog.getSaveFileName(self, "Save as JPEG", suggested, "JPEG Images (*.jpg)")
        if not destination:
            return
        try:
            self.service.save_jpeg(path, destination)
        except Exception as exc:  # noqa: BLE001 - surfaced to the photographer, not fatal
            QMessageBox.warning(self, "Save as JPEG", f"Could not save JPEG:\n{exc}")
            return
        QMessageBox.information(self, "Save as JPEG", f"Saved to {destination}")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override signature
        key = event.key()
        if key == Qt.Key.Key_K:
            self._apply_status("keep")
        elif key == Qt.Key.Key_R:
            self._apply_status("reject")
        elif key == Qt.Key.Key_N:
            self._apply_status("neutral")
        elif key == Qt.Key.Key_Right:
            self._go_next()
        elif key == Qt.Key.Key_Left:
            self._go_prev()
        else:
            super().keyPressEvent(event)
