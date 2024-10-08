import os
import math

class self_sort():
    
    def __init__(self,nums) -> None:
        self.nums = nums
    
    def max_heapify(self,begin,end):
        dad = begin
        son = begin*2+1
        while son<=end:
            if son+1 < end and self.nums[son] < self.nums[son+1]: son+=1
            if self.nums[dad] >= self.nums[son]:   
                return
            else:
                self.nums[dad],self.nums[son] = self.nums[son],self.nums[dad]
                dad = son
                son = dad*2+1
            
    def heapify(self):
        n = len(self.nums)
        for i in range(n//2-1,-1,-1):
            self.max_heapify(i,n)
        for i in range(n-1,0,-1):
            nums[0],nums[i] = nums[i], nums[0]
            self.max_heapify(0,i-1)
        return self.nums
    
    def quick_sort(self,start,end):
        if start>=end:
            return
        n = len(self.nums)
        l,r = start,end
        tmp = self.nums[start]
        while l < r:
            while l<r and self.nums[r]>=tmp:
                r -= 1
            if l < r:
                self.nums[l] = self.nums[r]
            while l < r and self.nums[l] <= tmp:
                l +=1
            if l < r:
                self.nums[r] = self.nums[l]
        nums[l] = tmp
        self.quick_sort(start,l-1)
        self.quick_sort(l+1,end)
    def quick_sort_use(self):
        self.quick_sort(0,len(self.nums)-1)
        return self.nums
    
    def bubble_sort(self):
        n = len(self.nums)
        idx = 0
        for i in range(n-idx):
            for j in range(i+1,n):
                if nums[i] >= nums[j]:
                    nums[i],nums[j] = nums[j],nums[i]
            idx +=1
        return self.nums   
    
    def inserted_sort(self):
        n = len(nums)
        # j = 0
        for i in range(n):
            tmp = nums[i]
            j = i-1
            while j >= 0 and nums[j] > tmp:
                nums[j+1] = nums[j]
                j -= 1
            nums[j+1] = tmp
        return self.nums
                
    def selected_sort(self):
        n = len(self.nums)
        for i in range(n-1):
            idx = i
            for j in range(i+1,n):
                if nums[j] > nums[idx]:
                    idx = j
            nums[idx],nums[i] = nums[i],nums[idx]
        return self.nums
                    
                    
if __name__ == "__main__":
    # nums = [0,3,1,2,4,6,6,10,5]
    nums = [3, 5, 3, 0, 8, 6, 1, 5, 8, 6, 2, 4, 9, 4, 7, 0, 1, 8, 9, 7, 3, 1, 2, 5, 9, 7, 4, 0, 2, 6]
    selfSort = self_sort(nums)
    # sort_nums = selfSort.heapify()
    # sort_nums = selfSort.quick_sort_use()
    # sort_nums = selfSort.bubble_sort()
    # sort_nums = selfSort.inserted_sort()
    sort_nums = selfSort.selected_sort()
    print(sort_nums)
    