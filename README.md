# Vk-analytics 
<h2> 
В данном репозитории я собрал небольшую работу, которую я сделал ради интереса, никакого глобального смысла в ней нет.</h2> 

Изначально я хотел попробовать по комментарию поста предсказать группу, из которой он пришел, но в процессе обработки и проверки нескольких алгоритмов, я понял, что работать он будет хуже, чем рандомный генератор.

Поэтому, я решил изменить стратегию и проделать аналогичную работу, используя чуть больше признаков.
Я все так же спарсил комментарии с каждого поста, но вместо того, чтобы по ним что-то предсказывать, я объединил все комментарии под каждым постом и кластеризировал их. Позже я использовал кластер как дополнительный категориальный признал для обучения классификатора.

<h6>Небольшой гайд по репозиторию:</h6>

parsing groups: парсер постов, парсил с помощью VK api

exploratory data analysis vk (2).ipynb: ноутбук с основной предобработкой и визуализацией данных по постам

comments clusterization.ipynb: nlp обработка комментариев и дальнейшая их кластеризация

VK Group label prediction.ipynb: итоговый файл с предсказанием групп
