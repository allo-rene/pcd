import torch
import torch.nn as nn
import torch.nn.functional as F


class PCD(nn.Module):
    def __init__(self, student, teacher, output_dim, queue_size, temperature):
        super().__init__()
        self.student = student
        self.teacher = teacher

        # freeze the teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        # create the memory queue
        self.register_buffer("queue", torch.randn(output_dim, queue_size))  # size of (o, k)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float))

    def _pixelwise_contrastive_loss(self, student_output, teacher_output):
        """
        first perform l2-normalize
        online and teacher: N, C, H, W
        """
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

        # dequeue and enqueue
        self._dequeue_and_enqueue(key_for_update)

        return {"loss": loss}
