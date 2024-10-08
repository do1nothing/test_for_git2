import os

def polygan_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i+1)%n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area)/2

if __name__ == "__main__":
    points = [[0,0],
              [1,0],
              [1,1],
              [0,1],
              [-0.5,0.5]]
    print(f"area = {polygan_area(points)}")
    a= 3
    print(a*a)
    print(a**3)
    print(pow(a,4))