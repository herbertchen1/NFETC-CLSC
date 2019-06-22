# NFETC-CLSC

Improving Distantly-supervised Entity Typing with Compact Latent Space Clustering

Paper accepted by NAACL-HLT 2019: [NFETC-CLSC]( https://arxiv.org/pdf/1904.06475.pdf)

### Prerequisites
- python 3.6.0
- tensorflow == 1.6.0
- hyperopt
- gensim
- sklearn
- pandas

Run `pip install -r requirement.txt` to satisfy the prerequisites.


### Dataset

Run `./download.sh` to download the pre-trained word embeddings.

The preprocessed dataset can be download from  [Google Drive](https://drive.google.com/open?id=1opjfoA0I2mOjE11kM_TYsaeq-HHO1rqv)

Put the data under the `./data/` directory

### Evaluation

Run `python eval.py -m <model_name> -d <data_name> -r <runs> -p <number> -a <alpha>`

The scores for each run and the average scores are also recorded in one log file stored in folder `log`

Available `<data_name>`:  ` bbn , ontonotes`

Available `<model_name>`:  ` nfetc_bbn_NFETC_CLSC , nfetc_ontonotes_NFETC_CLSC`

(which can be modified in `model_param_space.py`, the detailed hyper-parameter is in this file too)

Available `<number>` for noisy data: `5, 10, 15, 20, 25, 100`

Available `<number>` for clean data: `500, 1000, 1500, 2000, 2500` (You need to prepare the training file as mentioned before)

`<alpha>` is the hierarchy loss factor, `default == 0.0 `


### Cite

If you found this codebase or our work useful, please cite:

```
@inproceedings{chen-etal-2019-improving,
    title = "Improving Distantly-supervised Entity Typing with Compact Latent Space Clustering",
    author = "Chen, Bo  and
      Gu, Xiaotao  and
      Hu, Yufeng  and
      Tang, Siliang  and
      Hu, Guoping  and
      Zhuang, Yueting  and
      Ren, Xiang",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1294",
    pages = "2862--2872",
}
```

Note:

This code is based on the previous work by [Peng Xu](https://github.com/billy-inn). Many thanks to [Peng Xu](https://github.com/billy-inn).

Sincerely thanks [Konstantinos Kamnitsas](https://github.com/Kamnitsask)

for the guidance of the CLSC impelementation and the advice for the paper writting.
