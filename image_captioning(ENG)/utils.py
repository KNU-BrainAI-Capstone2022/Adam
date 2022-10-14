import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


train_print = 1000
valid_print = 100

#train mode
def train(train_loader, encoder, decoder, criterion, optimizer, vocab_size,
          epoch, total_step, start_step=1, start_loss=0.0):


    encoder.train()
    decoder.train()


    total_loss = start_loss


    start_train_time = time.time()

    for i_step in range(start_step, total_step + 1):

        indices = train_loader.dataset.get_indices()

        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_loader.batch_sampler.sampler = new_sampler


        for batch in train_loader:
            images, captions = batch[0], batch[1]
            break

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        features = encoder(images)
        outputs = decoder(features, captions)


        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


        stats = "Epoch %d, Train step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f" \
                % (epoch, i_step, total_step, time.time() - start_train_time,
                   loss.item(), np.exp(loss.item()))

        print("\r" + stats, end="")
        sys.stdout.flush()


        if i_step % train_print == 0:
            print("\r" + stats)
            filename = os.path.join("C:/Users/PC/Desktop/COCO/models", "train-model-{}{}.pkl".format(epoch, i_step))
            save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, i_step)
            start_train_time = time.time()

    return total_loss / total_step

#validation mode
def validate(val_loader, encoder, decoder, criterion, vocab, epoch,
             total_step, start_step=1, start_loss=0.0, start_bleu=0.0):



    encoder.eval()
    decoder.eval()


    smoothing = SmoothingFunction()


    total_loss = start_loss
    total_bleu_4 = start_bleu


    start_val_time = time.time()


    with torch.no_grad():
        for i_step in range(start_step, total_step + 1):

            indices = val_loader.dataset.get_indices()

            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            val_loader.batch_sampler.sampler = new_sampler


            for batch in val_loader:
                images, captions = batch[0], batch[1]
                break


            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()


            features = encoder(images)
            outputs = decoder(features, captions)


            batch_bleu_4 = 0.0


            for i in range(len(outputs)):
                predicted_ids = []
                for scores in outputs[i]:

                    predicted_ids.append(scores.argmax().item())

                predicted_word_list = word_list(predicted_ids, vocab)
                caption_word_list = word_list(captions[i].cpu().numpy(), vocab)

                batch_bleu_4 += sentence_bleu([caption_word_list],
                                              predicted_word_list,
                                              smoothing_function=smoothing.method1)
            total_bleu_4 += batch_bleu_4 / len(outputs)


            loss = criterion(outputs.view(-1, len(vocab)), captions.view(-1))
            total_loss += loss.item()


            stats = "Epoch %d, Val step [%d/%d], %ds, Loss: %.4f, Perplexity: %5.4f, Bleu-4: %.4f" \
                    % (epoch, i_step, total_step, time.time() - start_val_time,
                       loss.item(), np.exp(loss.item()), batch_bleu_4 / len(outputs))


            print("\r" + stats, end="")
            sys.stdout.flush()


            if i_step % valid_print == 0:
                print("\r" + stats)
                filename = os.path.join("C:/Users/PC/Desktop/COCO/models", "val-model-{}{}.pkl".format(epoch, i_step))
                save_val_checkpoint(filename, encoder, decoder, total_loss, total_bleu_4, epoch, i_step)
                start_val_time = time.time()

        return total_loss / total_step, total_bleu_4 / total_step


def save_checkpoint(filename, encoder, decoder, optimizer, total_loss, epoch, train_step=1):

    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "total_loss": total_loss,
                "epoch": epoch,
                "train_step": train_step,
                }, filename)


def save_val_checkpoint(filename, encoder, decoder, total_loss,
                        total_bleu_4, epoch, val_step=1):

    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "total_loss": total_loss,
                "total_bleu_4": total_bleu_4,
                "epoch": epoch,
                "val_step": val_step,
                }, filename)


def save_epoch(filename, encoder, decoder, optimizer, train_losses, val_losses,
               val_bleu, val_bleus, epoch):

    torch.save({"encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_bleu": val_bleu,
                "val_bleus": val_bleus,
                "epoch": epoch
                }, filename)

# 3epoch 동안 bleu score 향상되지 않으면 중단
def early_stopping(val_bleus, patience=3):

    if patience > len(val_bleus):
        return False
    latest_bleus = val_bleus[-patience:]

    if len(set(latest_bleus)) == 1:
        return True
    max_bleu = max(val_bleus)
    if max_bleu in latest_bleus:

        if max_bleu not in val_bleus[:len(val_bleus) - patience]:
            return False
        else:
            return True

    return True


def word_list(word_idx_list, vocab):

    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list


def clean_sentence(word_idx_list, vocab):

    sentence = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            sentence.append(word)
    sentence = " ".join(sentence)
    return sentence

# image captioning test
def get_prediction(data_loader, encoder, decoder, vocab):

    orig_image, image = next(iter(data_loader))
    plt.imshow(np.squeeze(orig_image))
    plt.title("Sample Image")
    plt.show()
    if torch.cuda.is_available():
        image = image.cuda()
    features = encoder(image).unsqueeze(1)
    print("Caption without beam search:")
    output = decoder.sample(features)
    sentence = clean_sentence(output, vocab)
    print(sentence)

    print("Top captions using beam search:")
    outputs = decoder.sample_beam_search(features)

    num_sents = min(len(outputs), 3)
    for output in outputs[:num_sents]:
        sentence = clean_sentence(output, vocab)
        print(sentence)