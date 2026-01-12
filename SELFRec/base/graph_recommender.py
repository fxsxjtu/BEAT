from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
import sys


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = []
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        for user in self.data.test_set:
            line = user + ':'
            for item in rec_list[user]:
                line += ' (' + item[0] + ',' + str(item[1]) + ')'
                if item[0] in self.data.test_set[user]:
                    line += '*'
            line += '\n'
            self.recOutput.append(line)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        out_dir = self.output['-dir']
        # file_name = self.config['model.name'] + '@' + current_time + '-top-' + str(self.max_N) + 'items' + '.txt'
        # FileIO.write_file(out_dir, file_name, self.recOutput)
        # print('The result has been output to ', abspath(out_dir), '.')
        file_name = self.config['model.name'] + '@' + current_time + '-' + self.config['dataset'] + '-performance' + '.txt'
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(self.result)
        FileIO.write_file(out_dir, file_name, self.result)
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def evaluateText(self, rec_list):
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        # output prediction result
        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # with open(f'/root/home/SELFRec/results/{self.args.prefix}.txt', 'a', encoding='utf-8') as file:
        #     file.write(str(self.result))
        #     file.write(str(self.pair_rate))
        #     file.write('\n')
        return self.result


    def fast_evaluation(self, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure

    def fast_evaluation_select(self, user_select, epoch):
        print('Evaluating the model...')
        rec_list = self.test()
        new_rec_list = {str(key): rec_list[str(key)] for key in set(user_select) if str(key) in rec_list}
        new_test_set = {str(key): self.data.test_set[str(key)] for key in set(user_select) if str(key) in self.data.test_set}
        measure = ranking_evaluation(new_test_set, new_rec_list, [self.max_N])
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            if count < 0:
                self.bestPerformance[1] = performance
                self.bestPerformance[0] = epoch + 1
                self.save()
        else:
            self.bestPerformance.append(epoch + 1)
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            self.bestPerformance.append(performance)
            self.save()
        print('-' * 120)
        print('Real-Time Ranking Performance ' + ' (Top-' + str(self.max_N) + ' Item Recommendation)')
        measure = [m.strip() for m in measure[1:]]
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
        bp = ''
        # for k in self.bestPerformance[1]:
        #     bp+=k+':'+str(self.bestPerformance[1][k])+' | '
        bp += 'Hit Ratio' + ':' + str(self.bestPerformance[1]['Hit Ratio']) + '  |  '
        bp += 'Precision' + ':' + str(self.bestPerformance[1]['Precision']) + '  |  '
        bp += 'Recall' + ':' + str(self.bestPerformance[1]['Recall']) + '  |  '
        # bp += 'F1' + ':' + str(self.bestPerformance[1]['F1']) + ' | '
        bp += 'NDCG' + ':' + str(self.bestPerformance[1]['NDCG'])
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance[0]) + ',', bp)
        print('-' * 120)
        return measure


def fast_evaluation(epoch, data, user_emb, item_emb, max_N):
    print('Evaluating the model...')
    rec_list = test(data, user_emb, item_emb, max_N)
    measure = ranking_evaluation(data.test_set, rec_list, [max_N])
    print('-' * 120)
    print('Real-Time Ranking Performance ' + ' (Top-' + str(max_N) + ' Item Recommendation)')
    measure = [m.strip() for m in measure[1:]]
    print('*Current Performance*')
    print('Epoch:', str(epoch + 1) + ',', '  |  '.join(measure))
    print(measure)
    print('-' * 120)
    return measure

import numpy as np
def predict(u, data, user_emb, item_emb):
    u = data.get_user_id(u)

    # 假设 self.user_emb 和 self.item_emb 是 NumPy 数组而不是 PyTorch 张量
    user_embedding = user_emb[u]  # 获取指定用户的嵌入
    item_embedding = item_emb.T  # 转置物品嵌入矩阵

    # 计算用户评分
    score = np.dot(user_embedding, item_embedding)  # 使用 numpy.dot 进行矩阵乘法
    return score

def test(data, user_emb, item_emb, max_N):
    def process_bar(num, total):
        rate = float(num) / total
        ratenum = int(50 * rate)
        r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
        sys.stdout.write(r)
        sys.stdout.flush()

    # predict
    rec_list = {}
    user_count = len(data.test_set)
    for i, user in enumerate(data.test_set):
        candidates = predict(user, data, user_emb, item_emb)
        # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
        rated_list, li = data.user_rated(user)
        for item in rated_list:
            candidates[data.item[item]] = -10e8
        ids, scores = find_k_largest(max_N, candidates)
        item_names = [data.id2item[iid] for iid in ids]
        rec_list[user] = list(zip(item_names, scores))
        if i % 1000 == 0:
            process_bar(i, user_count)
    process_bar(user_count, user_count)
    print('')
    return rec_list