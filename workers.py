import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, RapidOcrOptions, PdfBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

def gpu_worker(paper_chunk, gpu_id):
    device_str = f"cuda:{gpu_id}"
    
    acc_options = AcceleratorOptions(device=device_str, num_threads=4)
    pipe_options = ThreadedPdfPipelineOptions(
        accelerator_options=acc_options,
        ocr_batch_size=32,      
        pdf_backend=PdfBackend.PYPDFIUM2,
        do_ocr=True,
        ocr_options=RapidOcrOptions(backend="torch")
    )
    
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=pipe_options)})
    
    #store results as a list of objects instead of one big string
    processed_papers = []
    
    for paper in paper_chunk:
        try:
            print(f"GPU {gpu_id} reading: {paper['title'][:50]}...")
            result = converter.convert(paper['link'])
            content = result.document.export_to_markdown()
            
            #package the content with its citation metadata
            processed_papers.append({
                "title": paper['title'],
                "link": paper['link'],
                "source": paper.get('source', 'Unknown'),
                "text": content
            })
            torch.cuda.empty_cache() 
        except Exception as e:
            print(f"GPU {gpu_id} error on {paper['title'][:20]}: {e}")
            continue
            
    return processed_papers
