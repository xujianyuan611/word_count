#  from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    return render(request, 'home.html')


def count(request):
    user_text = request.GET['text']
    total_count = len(request.GET['text'])
    total_dict = {}
    for word in user_text:
        if word not in total_dict:
            total_dict[word] = 1
        else:
            total_dict[word] += 1

    sort_word = sorted(total_dict.items(), key=lambda x:x[1], reverse=True)

    return render(request, 'count.html', {'count': total_count, 'text': user_text, 'dict': total_dict,
                                          'sortword': sort_word} )