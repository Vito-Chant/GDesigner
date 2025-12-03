# m*n map 0 1
# (x1,y1) -> (x2,y2)
from collections import deque


def map2graph(map: list[list[int]]) -> dict:
    m = len(map)
    n = len(map[0])
    graph = dict()
    for i in range(m):
        for j in range(n):
            if map[i][j] == 0:
                graph[(i, j)] = []
                if i - 1 >= 0:
                    if map[i - 1][j] == 0:
                        graph[(i, j)].append((i - 1, j))
                if j + 1 < n:
                    if map[i][j + 1] == 0:
                        graph[(i, j)].append((i, j + 1))
                if i + 1 < m:
                    if map[i + 1][j] == 0:
                        graph[(i, j)].append((i + 1, j))
                if j - 1 >= 0:
                    if map[i][j - 1] == 0:
                        graph[(i, j)].append((i, j - 1))
            #
    return graph


def solution(map: list[list[int]], source: (int, int), destination: (int, int)):
    # boundary check
    graph = map2graph(map)
    step = 0
    visited = set()

    if source == destination:
        return step

    q = deque([source])

    while q:
        length = len(q)
        step += 1
        for i in range(length):
            node = q.popleft()
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if neighbor == destination:
                        return step
                    q.append(neighbor)

    return step


# 0 0 0 0
# 1 0 1 0
# 1 0 0 0

if __name__ == '__main__':
    map = [[0, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]]
    # graph = map2graph(map)
    # print(graph)
    source = (0, 0)
    destination = (0, 3)
    print(solution(map, source, destination))
