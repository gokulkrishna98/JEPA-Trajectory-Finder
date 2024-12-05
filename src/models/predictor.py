from torch import nn

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)

        self.h = None
        self.c = None

    def set_hc(self, h, c):
        self.h = h
        self.c = c 
    
    def reset_hc(self):
        self.h = self.h.zero_() 
        self.c = self.c.zero_()

    def forward(self, action):
        self.h, self.c = self.lstm_cell(action, (self.h, self.c))
        return self.h