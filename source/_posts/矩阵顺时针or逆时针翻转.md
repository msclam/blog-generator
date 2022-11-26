---
title: 矩阵顺时针or逆时针翻转
date: 2022-11-26 15:46:26
tags:
- 矩阵翻转
categories:
- 刷题相关
---

### 矩阵顺时针 or 逆时针翻转

顺时针：先主对角线(左上角到右下角这一斜线)、再左右

顺时针：先次对角线(右上角到左下角这一斜线)，再左右

<!-- more -->

1 2 3   

8 9 4 ->  

7 6 5



顺时针

```c++
for(int i = 0; i < n; i++){  //按照主对角线翻转
    for(int j = 0; j < i; j++){
        swap(matrix[i][j], matrix[j][i]);
    }
}

for(int i = 0; i < n; i++){  // 每一行按照中点进行翻转
    for(int j = 0; j < n / 2; j++){
        swap(matrix[i][j], matrix[i][n-j-1]);
    }
}
```

逆时针

```c++
for(int i = 0; i < n; i++){  // 次对角线翻转
    for(int j = 0; j < n - i; j++){
        swap(matrix[i][j], matrix[n-j-1][n-i-1]);
    }
}

for(int i = 0; i < n; i++){  // 每行按照中点翻转
    for(int j = 0; j < n / 2; j++){
        swap(matrix[i][j], matrix[i][n-j-1]);
    }
}
```
