# Dataset Preparation

## Training data

- annotation: We place the annotation file in [dataset/trainval/train_2k5.json](../dataset/trainval/train_2k5.json).

- original videos: You can download download from our organized version in [Boshenxx/TimeR1-Dataset](https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset); 

or you can download and organize the original data from [VTG-IT](https://huggingface.co/datasets/Yongxin-Guo/VTG-IT), [TimeIT](https://huggingface.co/datasets/ShuhuaiRen/TimeIT), [TimePro](https://huggingface.co/Lanxingxuan/TimeSuite), [HTStep](https://openreview.net/pdf?id=vv3cocNsEK) and [LongVid](https://huggingface.co/datasets/OpenGVLab/LongVid).


## Testing data


folder structure:
```
dataset                                                                           
├─ timer1          
│  ├─ annotations          
│  │  ├─ train_2k5.json                                                              
│  │  └─ tvgbench.json    
│  ├─ videos                                                                      
│  │  ├─ timerft_data                                                                 
│  │  |  ├─ xxx.mp4       
│  │  │  └─ ...
│  │  ├─ tvgbench_data                                                                      
│  │  |  ├─ xxx.mp4      
│  │  │  └─ ...
├─ activitynet                                                                    
│  ├─ annotations                                                                 
│  │  ├─ sentence_temporal_grounding                                              
│  │  │  └─ test.json                                                             
│  ├─ videos                                                                      
│  |  ├─ v_zzz_3yWpTXo.mp4       
│  │  └─ ...
├─ charades                                                                       
│  ├─ Charades_anno                                                               
│  │  └─ Charades_v1_test.csv                                                     
│  ├─ Charades_v1                                                                 
│  |  ├─ 0I0FX.mp4    
│  │  └─ ...
├─ mvbench                                       
│  ├─ json                                          
│  │  ├─ action_antonym.json                        
│  │  └─ ...               
│  ├─ videos                                       
│  │  ├─ clevrer
│  │  └─ ...
├─ tempcompass                                       
│  ├─ questions                                      
│  │  ├─ multi-choice.json                        
│  │  └─ ...               
│  ├─ videos                                       
│  │  ├─ 315784.mp4
│  │  └─ ...
├─ egoschema                                       
│  ├─ MC                                           
│  │  └─ test-00000-of-00001.parquet               
│  ├─ Subset                                       
│  │  └─ test-00000-of-00001.parquet               
│  ├─ videos                                       
│  │  ├─ 001934bb-81bd-4cd8-a574-0472ef3f6678.mp4  
│  │  └─ ...
├─ videomme                                       
│  ├─ videomme                                           
│  │  └─ test-00000-of-00001.parquet               
│  ├─ data                                       
│  │  ├─ _8lBR0E_Tx8.mp4     
└─ └─ └─ ...                                                          
```

### ActivityNet
Download link: [ActivityNet](https://cs.stanford.edu/people/ranjaykrishna/densevid/) 

For fine-tuning setting, before training, you need to preprocess the video data.

```bash
bash preprocess_video.sh
```
Specify the path to the Charades-STA dataset (video files, annotations, etc.).


### Charades
Download link: [Charades-v1](https://huggingface.co/datasets/HuggingFaceM4/charades)

For fine-tuning setting, before training, you need to preprocess the video data.

```bash
bash preprocess_video.sh
```
Specify the path to the Charades-STA dataset (video files, annotations, etc.).


### TVGBench
Download link: [hf: Boshenxx/TimeR1-Dataset](https://huggingface.co/datasets/Boshenxx/TimeR1-Dataset)


### MVBench

We place the annotation file of tvgbench in [dataset/trainval/tvgbench.json](../dataset/trainval/tvgbench.json).

Download link: [hf: OpenGVLab/MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench)


### VideoMME
Download link: [hf: lmms-lab/MVBench](https://huggingface.co/datasets/lmms-lab/Video-MME)


### Egoschema
Download link: [hf: lmms-lab/egoschema](https://huggingface.co/datasets/lmms-lab/egoschema)


### TempCompass

Download link: [hf: lmms-lab/TempCompass](https://huggingface.co/datasets/lmms-lab/TempCompass)

