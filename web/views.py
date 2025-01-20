from django.shortcuts import render
import torch
from .model.models import SLCABG
from .model.data_loader import process_data, get_word2index, case_folding, cleansing, set_stop_words
# Create your views here.
def predict(new_input):
    EMBED_SIZE = 768    
    SENTENCE_LENGTH = 12
    WORD_SIZE = 35000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sentences, label, word_vectors, word2index = process_data(SENTENCE_LENGTH, WORD_SIZE, EMBED_SIZE)
    net = SLCABG(EMBED_SIZE, SENTENCE_LENGTH, word_vectors).to(device)
    net.load_state_dict(torch.load('web/model/data/sentiment_analysis_fix.pth'))
    net.eval()
    with torch.no_grad():
        
        word2index = get_word2index()
        new_input = case_folding(new_input)
        new_input = cleansing(new_input)
        new_input = set_stop_words(new_input)
        new_input = new_input.split()
        new_input = [word2index[word] for word in new_input]
        if len(new_input) < SENTENCE_LENGTH:
            new_input.extend([0 for _ in range(SENTENCE_LENGTH - len(new_input))])
        else:
            new_input = new_input[:SENTENCE_LENGTH]
        new_input = torch.tensor(new_input).type(torch.LongTensor).to(device)
        new_input = new_input.unsqueeze(0)
        out = net(new_input)
        _, predicted = torch.max(out.data, 1)
        if predicted == 0:
            return 'positive'
        elif predicted == 1:
            return 'negative'
        else:
            return 'neutral'
        
def home(request):
    context = {}
    if request.method == 'POST':
        new_input = request.POST['new_input']
        result = predict(new_input)
        print(result)
        context['result'] = result
        
        return render(request, 'web/home.html', context)
    return render(request, 'web/home.html')