from pydantic import BaseModel
import pyvips
from PIL import Image
import io


from typing import List

from indexify_extractor_sdk import Extractor, Content, Feature, EmbeddingSchema, ExtractorSchema


class InputParams(BaseModel):
    a: int = 0
    b: str = ""


class MyExtractor(Extractor):
    def __init__(self):
        super().__init__()

    def extract(
        self, content: List[Content], params: InputParams
    ) -> List[List[Content]]:
        out = []
        for c in content:
            img = pyvips.Image.new_from_buffer(c.data, "")
            resized_img = img.resize(0.5)
            # target = pyvips.Target.new_to_memory()
            # resized_img.write_to_target(target, ".jpg")
            data = resized_img.write_to_buffer(".jpg")

            print(type(data))

            out.append([Content(content_type="image/jpeg",
                       data=data)])

        return out

    @classmethod
    def schemas(cls) -> ExtractorSchema:
        """
        Returns a list of options for indexing.
        """
        return ExtractorSchema(
            features={"embedding": EmbeddingSchema(
                distance_metric="cosine", dim=3)},
        )


if __name__ == "__main__":
    with open('rolex.jpg', 'rb') as file:
        image_bytes = file.read()
        e = MyExtractor()
        content = Content(content_type="image/jpeg",
                          data=image_bytes)
        # print(content)
        e.extract([content], InputParams())
