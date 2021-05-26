import torch
from config import CFG
from tqdm import tqdm

### Train one epoch

def train_text_fn(model, data_loader, optimizer, scheduler, use_sam, accum_iter, epoch, device, use_amp):
    model.train()
    
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Training epoch: " + str(epoch+1), ncols=100)

    for t, (texts, labels) in enumerate(tk):
        texts = list(texts)

        if use_sam:
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(texts, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                fin_loss += loss.item() 
                with torch.cuda.amp.autocast():
                     _, loss_second = model(texts, labels)
                loss_second.mean().backward()
                optimizer.second_step(zero_grad=True)
                optimizer.zero_grad()
            else:
                _, loss = model(texts, labels)
                loss.mean().backward()
                optimizer.first_step(zero_grad=True)
                fin_loss += loss.item() 
                _, loss_second = model(texts, labels)
                loss_second.mean().backward()
                optimizer.second_step(zero_grad=True)
                optimizer.zero_grad()

        else:  # if use_sam == False
            if use_amp:
                with torch.cuda.amp.autocast():
                    _, loss = model(texts, labels)
                scaler.scale(loss).backward()
                fin_loss += loss.item() 
                # mini-batch accumulation
                if (t + 1) % accum_iter == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                _, loss = model(texts, labels)
                loss.backward()
                fin_loss += loss.item() 
                # mini-batch accumulation
                if (t + 1) % accum_iter == 0:
                    optimizer.step() 
                    optimizer.zero_grad()
                
        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})

    scheduler.step()
    return model, fin_loss / len(data_loader)
