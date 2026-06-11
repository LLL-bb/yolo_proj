from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckoutPolygon:
    """A convex-or-concave polygon defining the checkout zone in pixel coordinates.

    Vertices are ordered (clockwise or CCW) and define a closed region.
    Uses the ray-casting algorithm for point-in-polygon tests.
    """

    vertices: tuple[tuple[int, int], ...]
    label: str = "checkout_zone"

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test.  Returns True for (x, y) inside the zone."""
        inside = False
        n = len(self.vertices)
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def scale_to(self, width: int, height: int, orig_w: int, orig_h: int) -> CheckoutPolygon:
        """Return a new polygon scaled from (orig_w × orig_h) to (width × height)."""
        sx, sy = width / orig_w, height / orig_h
        new_vertices = tuple((round(x * sx), round(y * sy)) for x, y in self.vertices)
        return CheckoutPolygon(vertices=new_vertices, label=self.label)


# ── Example: checkout zone for a static top-down-ish 1280×720 camera ───────
# Adjust these coordinates to match your camera setup.
DEFAULT_POLYGON = CheckoutPolygon(
    vertices=(
        (400, 520),
        (880, 520),
        (880, 640),
        (400, 640),
    ),
    label="checkout_basket",
)
