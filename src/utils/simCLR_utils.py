import torch
import torch.nn.functional as F


def nce_supervised_easy(out_1, out_2, label):
    out = torch.cat([out_1, out_2], dim=0)
    label = torch.cat([label, label], dim=0)
    cost = torch.exp(2 * torch.mm(out, out.t().contiguous()))
    batch = label.shape[0]
    pos_index = torch.zeros((batch, batch)).cuda()
    pos_index_2 = torch.zeros((batch, batch)).cuda()
    same_index = torch.eye(batch).cuda()
    for i in range(batch):
        ind = torch.where(label == label[i])[0]
        pos_index[i][ind] = 1
        if i < batch // 2:
            pos_index_2[i][i + batch // 2] = 1
            pos_index_2[i + batch // 2][i] = 1
    neg_index = 1 - pos_index
    pos_index = pos_index - same_index
    # print((pos_index).shape, (cost).shape)
    pos = pos_index * cost
    neg = neg_index * cost
    neg_exp_sum = (neg.sum(1)) / neg_index.sum(1)
    Nce = pos_index_2 * (pos / (pos + (batch - 2) * neg_exp_sum.reshape(-1, 1)))
    final_index = torch.where(Nce != 0)
    # print(pos_index[0].sum())
    # print(len(pos[0]), pos_index.sum())
    Nce = -((torch.log(Nce[final_index[0], final_index[1]])).mean())
    return Nce


def naive_contrastive_loss(feat_pos, feat_neg, temperature=0.5):
    feat_pos = F.normalize(feat_pos, dim=1)
    feat_neg = F.normalize(feat_neg, dim=1)

    # all ones to pos and all negative to neg
    l_pos = torch.ones((feat_pos.shape[0], 1), dtype=feat_pos.dtype).to(feat_pos.device)
    l_neg = -torch.ones((feat_neg.shape[0], 1), dtype=feat_neg.dtype).to(
        feat_neg.device
    )

    stack_features = torch.vstack([feat_pos, feat_neg])
    stack_labels = torch.vstack([l_pos, l_neg])

    similarity_matrix = torch.matmul(stack_features, stack_features.T)
    # label_matrix = ((2 - (torch.matmul(stack_labels, stack_labels.T) + 1)) / 2).to(
    #     dtype=similarity_matrix.dtype
    # )

    label_matrix = ((torch.matmul(stack_labels, stack_labels.T) + 1) / 2).to(
        dtype=similarity_matrix.dtype
    )

    # removing the diagonal
    mask = torch.eye(label_matrix.shape[0], dtype=torch.bool).to(label_matrix.device)

    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    label_matrix = label_matrix[~mask].view(label_matrix.shape[0], -1)

    # # computing logits and labels
    # positives = similarity_matrix[label_matrix.bool()].view(label_matrix.shape[0], -1)

    # l_positives = label_matrix[label_matrix.bool()].view(label_matrix.shape[0], -1)

    # negatives = similarity_matrix[~label_matrix.bool()].view(
    #     similarity_matrix.shape[0], -1
    # )

    # l_negatives = label_matrix[~label_matrix.bool()].view(label_matrix.shape[0], -1)

    # logits = torch.cat([positives, negatives], dim=1)
    # labels = torch.cat([l_positives, l_negatives], dim=1)
    # logits = logits / self.args.temperature

    return logits, labels


def info_nce_loss(self, features):
    labels = torch.cat(
        [torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0
    )
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)

    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    logits = logits / self.args.temperature
    return logits, labels


# class SimCLR(object):

#     def __init__(self, *args, **kwargs):
#         self.args = kwargs['args']
#         self.model = kwargs['model'].to(self.args.device)
#         self.optimizer = kwargs['optimizer']
#         self.scheduler = kwargs['scheduler']
#         self.writer = SummaryWriter()
#         logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
#         self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)


#     def train(self, train_loader):

#         scaler = GradScaler(enabled=self.args.fp16_precision)

#         # save config file
#         save_config_file(self.writer.log_dir, self.args)

#         n_iter = 0
#         logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
#         logging.info(f"Training with gpu: {self.args.disable_cuda}.")

#         for epoch_counter in range(self.args.epochs):
#             for images, _ in tqdm(train_loader):
#                 images = torch.cat(images, dim=0)

#                 images = images.to(self.args.device)

#                 with autocast(enabled=self.args.fp16_precision):
#                     features = self.model(images)
#                     logits, labels = self.info_nce_loss(features)
#                     loss = self.criterion(logits, labels)

#                 self.optimizer.zero_grad()

#                 scaler.scale(loss).backward()

#                 scaler.step(self.optimizer)
#                 scaler.update()

#                 if n_iter % self.args.log_every_n_steps == 0:
#                     top1, top5 = accuracy(logits, labels, topk=(1, 5))
#                     self.writer.add_scalar('loss', loss, global_step=n_iter)
#                     self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
#                     self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
#                     self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

#                 n_iter += 1

#             # warmup for the first 10 epochs
#             if epoch_counter >= 10:
#                 self.scheduler.step()
#             logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

#         logging.info("Training has finished.")
#         # save model checkpoints
#         checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
#         save_checkpoint({
#             'epoch': self.args.epochs,
#             'arch': self.args.arch,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#         }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
#         logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
