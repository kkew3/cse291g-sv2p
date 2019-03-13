import torch
import torch.nn.functional as F

from sv2p.cdna import ConditionalUNetLSTM
from sv2p.cdna import depthwise_conv2d
from sv2p.cdna import CDNA


# noinspection PyMethodMayBeStatic
class TestConditionalUNetLSTM:
    B = 17  # batch_size

    def test_shape_no_condition(self):
        net = ConditionalUNetLSTM(3, 0)
        inputs = self.init_inputs()

        embeddings, outputs, hidden = net(inputs[0])
        assert embeddings.size() == (self.B, 128, 8, 8)
        assert outputs.size() == (self.B, 64, 64, 64)
        _ = (torch.sum(embeddings) + torch.sum(outputs)).backward()

        _, _, hidden = net(inputs[0])
        embeddings2, outputs2, hidden = net(inputs[1], hidden_states=hidden)
        assert embeddings2.size() == (self.B, 128, 8, 8)
        assert outputs2.size() == (self.B, 64, 64, 64)
        _ = (torch.sum(embeddings2) + torch.sum(outputs2)).backward()

    def test_shape_with_condition(self):
        net = ConditionalUNetLSTM(3, 2)
        inputs, conds = self.init_inputs(condition=True)

        embeddings, outputs, hidden = net(inputs[0], conditions=conds[0])
        assert embeddings.size() == (self.B, 128, 8, 8)
        assert outputs.size() == (self.B, 64, 64, 64)
        _ = (torch.sum(embeddings) + torch.sum(outputs)).backward()

        _, _, hidden = net(inputs[0], conditions=conds[0])
        embeddings2, outputs2, hidden = net(inputs[1], conditions=conds[1],
            hidden_states=hidden)
        assert embeddings2.size() == (self.B, 128, 8, 8)
        assert outputs2.size() == (self.B, 64, 64, 64)
        _ = (torch.sum(embeddings2) + torch.sum(outputs2)).backward()


    def init_inputs(self, condition=False):
        ret = [torch.rand(self.B, 3, 64, 64) for _ in range(2)]
        if condition:
            conds = [torch.rand(self.B, 2, 8, 8) for _ in range(2)]
            ret = (ret, conds)
        return ret


def test_depthwise_conv2d():
    for _ in range(10):
        imgs = torch.rand(16, 3, 64, 64)
        kernel = torch.rand(10, 1, 5, 5)

        expected = torch.stack([F.conv2d(imgs[:, i:i+1], kernel, padding=2)
                                for i in range(3)], dim=2)
        actual = depthwise_conv2d(imgs, kernel, padding=2)
        assert expected.size() == (16, 10, 3, 64, 64)
        assert actual.size() == (16, 10, 3, 64, 64)
        assert torch.allclose(actual, expected)


class TestCDNA:
    def test_shape_no_condition(self):
        net = CDNA(3, 0, 10)
        inputs = [torch.rand(16, 3, 64, 64) for _ in range(2)]

        predictions, hidden, _, _ = net(inputs[0])
        assert predictions.size() == (16, 3, 64, 64)
        _ = torch.sum(predictions).backward()

        _, hidden, _, _ = net(inputs[0])
        predictions2, hidden, _, _ = net(inputs[1], hidden_states=hidden)
        assert predictions2.size() == (16, 3, 64, 64)
        _ = torch.sum(predictions2).backward()

    def test_shape_with_condition(self):
        net = CDNA(3, 2, 10)
        inputs = [torch.rand(16, 3, 64, 64) for _ in range(2)]
        conds = [torch.rand(16, 2, 8, 8) for _ in range(2)]

        predictions, hidden, _, _ = net(inputs[0], conditions=conds[0])
        assert predictions.size() == (16, 3, 64, 64)
        _ = torch.sum(predictions).backward()

        _, hidden, _, _ = net(inputs[0], conditions=conds[0])
        predictions2, hidden, _, _ = net(inputs[1], conditions=conds[1],
                                   hidden_states=hidden)
        assert predictions2.size() == (16, 3, 64, 64)
        _ = torch.sum(predictions2).backward()
