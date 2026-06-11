from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    price_cents: int  # stored as integer cents to avoid float rounding
    category: str = "general"


# Maps YOLO class_id → Product.  IDs correspond to the filtered RPC dataset
# (76 classes: puffed_food 0-11, dessert 12-28, drink 29-45, chocolate 46-57,
#  gum 58-65, candy 66-75).
CATALOG: dict[int, Product] = {
    # Drinks (YOLO 29-45)
    29: Product(id="water_bottle",  name="Spring Water 550mL",     price_cents=129, category="beverages"),
    32: Product(id="coke_can_355",  name="Coca-Cola 500mL",        price_cents=199, category="beverages"),
    33: Product(id="pepsi_can_355", name="Pepsi 600mL",            price_cents=199, category="beverages"),
    36: Product(id="sprite_can",    name="Sprite 500mL",           price_cents=199, category="beverages"),
    34: Product(id="fanta_apple",   name="Fanta Apple 500mL",      price_cents=199, category="beverages"),

    # Snacks (YOLO 0-11 puffed_food)
    9:  Product(id="cheetos",       name="Cheetos Japanese Steak", price_cents=449, category="snacks"),

    # Dessert (YOLO 12-28)
    28: Product(id="oreo_pack",     name="Oreo Cookies 55g",       price_cents=399, category="snacks"),

    # Chocolate (YOLO 46-57)
    55: Product(id="snickers_bar",  name="Snickers Bar 51g",       price_cents=179, category="candy"),
}


class ProductCatalog:
    """Resolves class IDs and product IDs to Product objects."""

    @staticmethod
    def get_by_class_id(class_id: int) -> Product:
        product = CATALOG.get(class_id)
        if product is None:
            raise KeyError(f"No product mapping for class_id={class_id}")
        return product

    @staticmethod
    def get_by_id(product_id: str) -> Product:
        for product in CATALOG.values():
            if product.id == product_id:
                return product
        raise KeyError(f"No product with id='{product_id}'")

    @staticmethod
    def class_id_for(product_id: str) -> int:
        for cid, product in CATALOG.items():
            if product.id == product_id:
                return cid
        raise KeyError(f"No class_id found for product_id='{product_id}'")
