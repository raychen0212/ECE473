import shell
import util
import wordsegUtil

############################################################
# Problem 1: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.query
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if len(state) == 0:
            return True
        else:
            return False
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        expand_list = []

        for i in range(0, len(state) + 1):
            nxt_state = state[0:i]
            curr_state = state[i:len(state)+1]
            expand_list.append((nxt_state, curr_state, self.unigramCost(nxt_state)))

        return expand_list

        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 2: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def start(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)

        return (wordsegUtil.SENTENCE_BEGIN, 0)
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def goalp(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state[1] == len(self.queryWords)

        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def expand(self, state):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        expand_list = []
        all_possible = self.possibleFills(self.queryWords[state[1]])
        if len(all_possible) == 0:
            all_possible.add(self.queryWords[state[1]])
        for i in all_possible:
            remain = (i, state[1]+1)
            nxt_state = i
            cost = self.bigramCost(state[0], i)
            expand_list.append((nxt_state, remain, cost))
        return expand_list

        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)

    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))

    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    return ' '.join(ucs.actions)


if __name__ == '__main__':
    shell.main()
