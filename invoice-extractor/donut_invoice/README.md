# Simple Invoice Extractor

This extractor parses some invoice-related data from a PDF.
It uses the pre-trained [donut model from huggingface](https://huggingface.co/docs/transformers/model_doc/donut).

Content[String] -> Content[Empty] + Features[JSON metadata of invoice].

Example input:

```text
<PDF file of an Invoice>
```

Example output:

```json
{
    "invoice_simple_donut": {
        "DocType": "Invoice",
        "Currency1": "CHF",
        "DocumentDate": "2023-01-11",
        "GrossAmount": "269.25",
        "InvoiceNumber": "8037",
        "NetAmount1": "250.00",
        "TaxAmount1": "19.25"
    }
}
```

## Usage

Try out the extractor. Write your favorite (foreign) quote.

```bash
cd invoice-extractor
indexify extractor extract --file invoice.pdf
```

## Container

* The container is not published yet. *

```bash
docker run  -it yenicelik/simple-invoice-parser extractor extract --file invoice.pdf
```
