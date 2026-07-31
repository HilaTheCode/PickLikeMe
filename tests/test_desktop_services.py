from pathlib import Path

from picklikeme.desktop.services import ReviewService


def test_review_service_exposes_session_state_and_decisions(tmp_path: Path) -> None:
    image_path = tmp_path / "demo.jpg"
    image_path.write_bytes(b"fake image")

    service = ReviewService(db_path=tmp_path / "annotations.sqlite")
    try:
        state = service.open_folder(tmp_path)
        assert state["input_folder"] == str(tmp_path.resolve())
        assert any(item["filename"] == "demo.jpg" for item in state["images"])

        updated = service.set_review_status(str(image_path), "keep")
        assert updated == "keep"
        state = service.load_session()
        item = next(item for item in state["images"] if item["filename"] == "demo.jpg")
        assert item["review_status"] == "keep"
    finally:
        service.close()
