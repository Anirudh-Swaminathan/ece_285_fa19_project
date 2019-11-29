import torch,utils.data as d
from vocab_creator import VocabCreate as vc
from framework import COCODataset

#dict_path = "../outputs/vocab.npz"
#vocab = vc(train_ann, dict_path)

#dataset_root_dir = '/datasets/COCO-2015/'
#annotations_root_dir = '../datasets/COCO/annotations/'
#training_dataset = COCODataset(dataset_root_dir, annotations_root_dir, vocab)

class preprocessing(d.DataLoader):

    def __init__(self,dataset_root_dir,annotations_root_dir,dict_path,mode="train",batch_size=128,shuffle=True)
        self.mode_loc = "{}captions_{}.json".format(annotations_root_dir, mode)
        self.vocab=vc(self.mode_loc,dict_path)
        self.data=COCODataset(dataset_root_dir,annotations_root_dir,vocab)
    
    def __collate__(self):
        #to sort data by caption length
        self.data.sort(key=lambda x:len(x[1]),reverse=True)
        img,cap=zip(*self.data)

        #stack images
        img=torch.stack(img,0)

        #concatenate all captions
        cap_lens=[len(c) for c in cap]

        #pad all captions to max caption length
        padded_caption = torch.zeros(len(cap),max(cap_lens)).long()
        for i,c in enumerate(cap):
            c_len=cap_lens[i]
            padded_cap[i,:end] = c[:end]

        return img,padded_cap,cap_lens


    def __ prepdata__():
        prepped_data=torch.utils.data.DataLoader(dataset=self.data,batch_size=self.batch_size,
                shuffle=self.shuffle,collate_fn=collate)

        return prepped_data



test_obj=preprocessing(dataset_root_dir,annotations_root_dir,dict_path,mode="train")
