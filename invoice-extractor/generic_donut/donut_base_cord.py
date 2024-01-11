import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, DonutProcessor
from typing import List

class ImageContent:
    def __init__(self, image_path):
        self.content = image_path

class ImageExtractionParams:
    pass

class BaseExtractor:
    def __init__(self):
        pass

class InvoiceDataExtractor(BaseExtractor):
    def __init__(self):
        super(InvoiceDataExtractor, self).__init__()
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v1")
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v1")

    def extract(self, images: List[ImageContent], params: ImageExtractionParams) -> List[dict]:
        extracted_data = []
        for image in images:
            try:
                image = Image.open(image.content).convert("RGB")
            except IOError:
                print(f"Error: Unable to open image file {image.content}.")
                continue

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            print(pixel_values.shape)
            output_ids = self.model.generate(pixel_values, max_new_tokens=50)
            print(output_ids.shape)
            print(output_ids)
            decoded_output = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(decoded_output)
            
            structured_data = self.post_process(decoded_output)
            extracted_data.append(structured_data)

        return extracted_data

    def post_process(self, text):
        structured_data = {}
        for line in text.split('\n'):
            parts = line.split(':')
            if len(parts) == 2:
                key, value = parts
                structured_data[key.strip()] = value.strip()
        return structured_data

if __name__ == "__main__":
    extractor = InvoiceDataExtractor()
    image_path = 'invoice-extractor\generic_donut\ex_sa_inv.jpg'
    image_content = ImageContent(image_path)
    extraction_params = ImageExtractionParams()
    extracted_data = extractor.extract([image_content], extraction_params)
    print(extracted_data)
