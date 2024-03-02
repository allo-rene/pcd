import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .utils import accuracy


class PCD(nn.Module):
    def __init__(self, student, teacher, output_dim, queue_size, temperature):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.queue_size = queue_size

        # freeze the teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # create the memory queue
        self.register_buffer("queue", torch.randn(output_dim, queue_size))  # size of (o, k)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float))

    def _pixelwise_contrastive_loss(self, student_output, teacher_output):
        N, C, H, W = student_output.shape
        student_output = F.normalize(student_output.view(N, C, -1), dim=1, p=2)
        teacher_output = F.normalize(teacher_output.detach().view(N, C, -1), dim=1, p=2)

        pos = (student_output * teacher_output).sum(dim=1).view(-1, 1)  # NHW, 1
        neg = torch.matmul(student_output.transpose(1, 2),
                           self.queue.clone().detach()).view(-1, self.queue_size)  # NHW, queue_size
        logit = torch.cat((pos, neg), dim=1).div(self.temperature)  # NHW, 1 + queue_size
        loss = F.cross_entropy(logit, torch.zeros(logit.size(0), dtype=torch.long).cuda())
        return loss

    def forward(self, image1, image2):
        student_output1 = self.student(image1)
        student_output2 = self.student(image2)

        with torch.no_grad():
            teacher_output1 = self.teacher(image1)
            teacher_output2 = self.teacher(image2)

            # for negative queue
            key_for_update = F.adaptive_avg_pool2d(teacher_output1, 1).view(teacher_output1.size(0), -1)
            key_for_update = F.normalize(key_for_update, dim=1, p=2)

        # loss
        loss = self._pixelwise_contrastive_loss(student_output1, teacher_output2) + \
               self._pixelwise_contrastive_loss(student_output2, teacher_output1)
        loss /= 2.

        # linear probing accuracy for monitoring
        with torch.no_grad():
            vec_student_output = student_output1.detach().clone().mean(dim=[2, 3])
            pos_logit = vec_student_output @ key_for_update.t()  # N, 1
            neg_logit = vec_student_output @ self.queue.clone().detach().t()  # N, K
            logit = torch.cat([pos_logit, neg_logit], dim=1)
            label = torch.zeros(logit.shape[0], dtype=torch.long, device=logit.device)
            top1, top5 = accuracy(logit, label, (1, 5))

        # dequeue and enqueue
        self._dequeue_and_enqueue(key_for_update)

        return {"loss": loss, "top1": top1, "top5": top5}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # this function is copied from MoCo.
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    This function is copied from MoCo.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
