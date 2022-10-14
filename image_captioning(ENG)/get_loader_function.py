def get_loader(transform,
               mode="train",
               batch_size=1,
               vocab_threshold=None,
               vocab_file="C:/Users/PC/Desktop/COCO/vocab/vocab.pkl",
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc="C:/Users/PC/Desktop/COCO"):
    
    assert mode in ["train", "val", "test"]
   

    
    if mode == "train":
        if vocab_from_file == True:
            assert os.path.exists(vocab_file), "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."
        else:
            assert mode == "train", "To generate vocab from captions file, must be in training mode (mode='train')."
                
        img_folder = os.path.join(cocoapi_loc, "data/train2017/")
        annotations_file = os.path.join(cocoapi_loc, "data/annotations/captions_train2017.json")
    
    if mode == "val":
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "data/val2017/")
        annotations_file = os.path.join(cocoapi_loc, "data/annotations/captions_val2017.json")
        
    if mode == "test":
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, "data/test2017/")
        annotations_file = os.path.join(cocoapi_loc, "data/annotations/image_info_test2017.json")

   
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == "train":
        
        indices = dataset.get_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader