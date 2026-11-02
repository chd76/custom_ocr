# Third-party components, licenses and citations

This pipeline is built on the software and models below. None of their code or model files is copied into this
repository except the fine-tuned page-filter model (see "Models"); everything else is installed by `setup/setup_envs.py`
or downloaded by the libraries on first use. Versions are the ones in `setup/lock-*.txt`.

## Software

| Component | Version | Used for | License | Source |
|---|---|---|---|---|
| PaddleOCR | 3.7.0 | PP-StructureV3 document parsing (stage 3) | Apache-2.0 | https://github.com/PaddlePaddle/PaddleOCR |
| PaddleX | 3.7.2 | the pipeline code under PaddleOCR (stage 3 patches its layout-parsing steps at run time) | Apache-2.0 | https://github.com/PaddlePaddle/PaddleX |
| PaddlePaddle (paddlepaddle-gpu) | 3.4.0 | deep learning framework for PaddleOCR | Apache-2.0 | https://github.com/PaddlePaddle/Paddle |
| Ultralytics | 8.4.177 | page filter: training and running the classifier (stage 2, `page_filter/`) | AGPL-3.0 | https://github.com/ultralytics/ultralytics |
| PyTorch / torchvision | 2.11.0 / 0.26.0 | framework for Ultralytics | BSD-3-Clause | https://github.com/pytorch/pytorch |
| NumPy | 2.x | arrays (used by the libraries above) | BSD-3-Clause | https://github.com/numpy/numpy |
| OpenCV (opencv-python / opencv-contrib-python) | 5.0 / 4.10 | image handling (used by the libraries above) | Apache-2.0 | https://github.com/opencv/opencv |
| Pillow | 13.0.0 | reading pages, drawing check pictures (stage 5), labeling tool | MIT-CMU | https://github.com/python-pillow/Pillow |
| PyYAML | 6.0 | settings | MIT | https://github.com/yaml/pyyaml |
| uv | – | creating the environments | Apache-2.0 / MIT | https://github.com/astral-sh/uv |

## Models

| Model | Used for | License | Source |
|---|---|---|---|
| PP-DocLayout_plus-L | layout blocks (text, titles, captions, ...) | Apache-2.0 | https://huggingface.co/PaddlePaddle/PP-DocLayout_plus-L |
| PP-DocBlockLayout | article areas (`use_region_detection`) | Apache-2.0 | https://huggingface.co/PaddlePaddle/PP-DocBlockLayout |
| PP-OCRv6_medium_det | text line detection | Apache-2.0 | https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det |
| PP-OCRv6_medium_rec | text recognition | Apache-2.0 | https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec |
| YOLO26n-cls (Ultralytics) | starting weights of the page filter | AGPL-3.0 | https://docs.ultralytics.com/models/yolo26/ |
| `page_filter/runs/page_filter/weights/best.pt` (included) | page filter: keep / mixed / trash; YOLO26n-cls fine-tuned on our own page labels | AGPL-3.0 (derived from YOLO26 weights) | this repository |

The PaddleOCR models are downloaded by PaddleOCR on first use; the PP-OCRv6 models were released with PaddleOCR 3.7.0
and have no separate technical report, so the PaddleOCR 3.0 Technical Report is cited for them.

## Changes to third-party code

`s3_paddle.py` replaces three functions of PaddleX's layout-parsing pipeline **at run time** (no PaddleX file is changed
or copied): the block sorting (to keep each block's article-area number, re-implementing PaddleX's own sorting),
`LayoutBlock.group_boxes_into_lines` (to drop bogus line pieces first) and the line-to-block matching in
`standardized_data` (each text line goes to one block). See the docstrings in `s3_paddle.py`.

## BibTeX

```bibtex
@misc{cui2025paddleocr30technicalreport,
      title={PaddleOCR 3.0 Technical Report},
      author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2507.05595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05595},
}

@misc{jocher2026ultralyticsyolo26unifiedrealtime,
      title={Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models},
      author={Glenn Jocher and Jing Qiu and Mengyu Liu and Shuai Lyu and Fatih Cagatay Akyon and Muhammet Esat Kalfaoglu},
      year={2026},
      eprint={2606.03748},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03748},
}

@misc{paszke2019pytorchimperativestylehighperformance,
      title={PyTorch: An Imperative Style, High-Performance Deep Learning Library},
      author={Adam Paszke and Sam Gross and Francisco Massa and Adam Lerer and James Bradbury and Gregory Chanan and Trevor Killeen and Zeming Lin and Natalia Gimelshein and Luca Antiga and Alban Desmaison and Andreas Köpf and Edward Yang and Zach DeVito and Martin Raison and Alykhan Tejani and Sasank Chilamkurthy and Benoit Steiner and Lu Fang and Junjie Bai and Soumith Chintala},
      year={2019},
      eprint={1912.01703},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1912.01703},
}

@misc{harris2020arrayprogrammingnumpy,
      title={Array Programming with NumPy},
      author={Charles R. Harris and K. Jarrod Millman and Stéfan J. van der Walt and Ralf Gommers and Pauli Virtanen and David Cournapeau and Eric Wieser and Julian Taylor and Sebastian Berg and Nathaniel J. Smith and Robert Kern and Matti Picus and Stephan Hoyer and Marten H. van Kerkwijk and Matthew Brett and Allan Haldane and Jaime Fernández del Río and Mark Wiebe and Pearu Peterson and Pierre Gérard-Marchant and Kevin Sheppard and Tyler Reddy and Warren Weckesser and Hameer Abbasi and Christoph Gohlke and Travis E. Oliphant},
      year={2020},
      eprint={2006.10256},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      doi={10.1038/s41586-020-2649-2},
      url={https://arxiv.org/abs/2006.10256},
}
```

PaddlePaddle, PaddleX, OpenCV, Pillow and PyYAML are cited as software (name, version, URL above).
