from __future__ import annotations

from typing import Optional

from config.polygon import CheckoutPolygon
from config.settings import settings
from core_logic.interfaces import ZoneChecker
from core_logic.models import TrackedObject, ZoneStatus


class PointInPolygonChecker(ZoneChecker):
    """Classifies tracked objects against a single polygon checkout zone.

    Uses the **bottom-center** of each bounding box by default (the point
    where an item contacts the counter/shelf).  This is more robust than the
    centroid for retail contexts.
    """

    def __init__(
        self,
        polygon: Optional[CheckoutPolygon] = None,
        point_mode: Optional[str] = None,
    ) -> None:
        self._polygon = polygon or CheckoutPolygon(vertices=((0, 0), (0, 0)))
        self._point_mode = point_mode or settings.point_mode

    def set_polygon(self, polygon: CheckoutPolygon) -> None:
        self._polygon = polygon

    def classify(self, obj: TrackedObject) -> ZoneStatus:
        x1, y1, x2, y2 = obj.bbox

        if self._point_mode == "centroid":
            px = (x1 + x2) / 2.0
            py = (y1 + y2) / 2.0
        else:  # "bottom_center"
            px = (x1 + x2) / 2.0
            py = y2

        return ZoneStatus.INSIDE if self._polygon.contains(px, py) else ZoneStatus.OUTSIDE
