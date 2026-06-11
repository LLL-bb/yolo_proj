import pytest

from config.polygon import CheckoutPolygon
from core_logic.models import TrackedObject, ZoneStatus
from core_logic.zone_checker import PointInPolygonChecker


class TestPointInPolygonChecker:
    """Verify zone classification with a simple rectangular polygon."""

    @pytest.fixture
    def checker(self) -> PointInPolygonChecker:
        polygon = CheckoutPolygon(
            vertices=((100, 100), (300, 100), (300, 300), (100, 300)),
            label="test_zone",
        )
        return PointInPolygonChecker(polygon=polygon, point_mode="bottom_center")

    def make_obj(self, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
        return TrackedObject(
            tracking_id=1,
            bbox=(x1, y1, x2, y2),
            class_id=0,
            class_name="test",
            confidence=1.0,
        )

    def test_inside_bottom_center(self, checker: PointInPolygonChecker) -> None:
        """Bottom-center (200, 200) is well inside the polygon."""
        obj = self.make_obj(150, 150, 250, 250)
        assert checker.classify(obj) is ZoneStatus.INSIDE

    def test_outside_below(self, checker: PointInPolygonChecker) -> None:
        """Bottom-center (200, 400) is below the polygon."""
        obj = self.make_obj(150, 350, 250, 450)
        assert checker.classify(obj) is ZoneStatus.OUTSIDE

    def test_outside_left(self, checker: PointInPolygonChecker) -> None:
        """Bottom-center (50, 200) is left of the polygon."""
        obj = self.make_obj(0, 150, 100, 250)
        assert checker.classify(obj) is ZoneStatus.OUTSIDE

    def test_bbox_on_boundary(self, checker: PointInPolygonChecker) -> None:
        """Bottom-center slightly above the bottom edge (avoids ray-casting epsilon)."""
        obj = self.make_obj(200, 290, 250, 299)  # bottom-center y=299 < 300
        assert checker.classify(obj) is ZoneStatus.INSIDE

    def test_centroid_mode(self) -> None:
        polygon = CheckoutPolygon(
            vertices=((100, 100), (300, 100), (300, 300), (100, 300)),
        )
        checker = PointInPolygonChecker(polygon=polygon, point_mode="centroid")
        obj = TrackedObject(
            tracking_id=1,
            bbox=(150, 150, 250, 250),
            class_id=0,
            class_name="test",
            confidence=1.0,
        )
        assert checker.classify(obj) is ZoneStatus.INSIDE
