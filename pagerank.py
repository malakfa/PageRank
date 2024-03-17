import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """    
    num_pages = len(corpus)
    probability_distribution = {}

    if not corpus[page]:
        for p in corpus:
            probability_distribution[p] = 1 / num_pages
    else:
       
        link_prob = damping_factor / len(corpus[page])
        uniform_prob = (1 - damping_factor) / num_pages

        for p in corpus:
            probability_distribution[p] = uniform_prob

        for linked_page in corpus[page]:
            probability_distribution[linked_page] += link_prob

    return probability_distribution



def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    current_sample = list(corpus.keys())[0] 
    page_ranks = {p: 0 for p in corpus}
    page_ranks[current_sample] = 1
    for i in range(n):
        probabilities  = transition_model(corpus , current_sample , damping_factor)

        next_page = random.choices(list(probabilities.keys()) , list(probabilities.values()))[0]

        page_ranks[next_page] = page_ranks[next_page] + 1

        current_sample = next_page

    for p in corpus:
        page_ranks[p] = page_ranks[p]/n

    return page_ranks



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    pagerank_distribution = {page: 1 / n for page in corpus}
    
    
    stop = False
    while stop == False:
        new_pagerank = {}
        for p in corpus :
            new_pagerank[p] = (1 - damping_factor)/n
            for linking_page, links in corpus.items():
                if p in links:
                    new_pagerank[p] += damping_factor * pagerank_distribution[linking_page] / len(links)

        
        counter = 0
        for p in corpus :
            if abs(new_pagerank[p] - pagerank_distribution[p]) <= 0.001 :
                counter += 1
        
        pagerank_distribution = new_pagerank.copy()
        if counter == n-1:
            stop = True
        
    return new_pagerank


if __name__ == "__main__":
    main()
