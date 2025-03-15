
#'input_output': '{"inputs": ["L = 2, points = {{1,0},{1,2}}", "L = 2.8284, points:{{1,1}, {-1,-1}}"], 
# "outputs": ["{{0,0},{0,2},{2,0},{2,2}}", "{{-2,0},{0,-2},{0,2},{2,0}}"]}
class SolutionCreated:
    def findCornerPoints(self, L, points):
        p = points[0]
        q = points[1]
        A = [p[0] - L/2, (p[1] + q[1])/2]
        B = [(p[0] + q[0])/2, q[1] - L/2]
        C = [(p[0] + q[0])/2, q[1] + L/2]
        D = [p[0] + L/2, (p[1] + q[1])/2]
        return sorted([A, B, C, D])
    
class CanonicalSolution:
    def findCornerPoints(self, L, points):
        import math
        y = points[1][0] - points[0][0]
        x = -(points[1][1] - points[0][1])
        k = math.sqrt(x ** 2 + y ** 2)
        x = L / 2.0 * x / k
        y = L / 2.0 * y / k
        result = []
        result.append([math.floor(points[0][0] - x), math.floor(points[0][1] - y)])
        result.append([math.floor(points[1][0] - x), math.floor(points[1][1] - y)])
        result.append([math.floor(points[0][0] + x), math.floor(points[0][1] + y)])
        result.append([math.floor(points[1][0] + x), math.floor(points[1][1] + y)])
        result = sorted(result)
        return result

s = SolutionCreated()
print(s.findCornerPoints(2, [[1, 0], [1, 2]]))
print(s.findCornerPoints(2.8284, [[1,1], [-1,1]])) # Expected: [[-1, 1], [0, -1], [0, 3], [1, 1]]
c= CanonicalSolution()
print(c.findCornerPoints(2, [[1, 0], [1, 2]])) 
print(c.findCornerPoints(2.8284, [[1,1], [-1,1]])) # Expected: [[-1, 1], [0, -1], [0, 3], [1, 1]]