---
title: hexo+fluid部署博客
date: 2022-11-25 21:07:15
tags:
- blog
categories:
- 工具相关
- hexo博客部署
---

hexo+github的初始化部署、发布新篇、更换主题、上传创建源码、一些小问题

<!-- more -->

### 1 hexo + github的部署

```
1、进入一个安全的目录，比如 cd ~/Desktop 或者 cd ~/Documents，不要根目录 /
2、在 GitHub 上新建一个空 repo，repo 名称是「你的用户名.github.io」（注意个用户名是你的GitHub用户名，不是你的电脑用户名）
3、npm install -g hexo-cli，安装 Hexo
4、hexo init myBlog
5、cd myBlog
6、npm i
7、hexo new 开博大吉，你会看到一个 md 文件的路径，Windows 的路径中的 \ 需要变成 / 才行哦
8、start x.md，编辑这个 md 文件，（Ubuntu 系统用 xdg-open x.md 命令）
9、start _config.yml，编辑网站配置
    （1）把第 6 行的 title 改成标题
    （2）把第 9 行的 author 改成你的名字
    （3）把最后一行的 type 改成 type: git
    （4）在最后一行后面新增一行，左边与 type 平齐，加上一行 repo: 仓库地址
    	（请将仓库地址改为「你的用户名.github.io」对应的仓库地址，仓库地址以 git@github.com: 开头）
    （5）第 4 步的 repo: 后面有个空格
10、npm install hexo-deployer-git --save，安装 git 部署插件
11、hexo deploy
12、进入「你的用户名.github.io」对应的 repo，打开 GitHub Pages 功能，如果已经打开了，就直接点击预览链接，可以在仓库右边点击edit保存page的部署网址
```



### 2 发布新篇

```
hexo n name
hexo clean && hexo generate && hexo deploy
或者简写
hexo cl && hexo g && hexo d
```



### 3 更换主题

```java
1、https://github.com/hexojs/hexo/wiki/Themes 上面有主题合集
2、随便找一个主题，进入主题的 GitHub 首页，比如我找的是 https://github.com/iissnan/hexo-theme-fluid
3、复制它的 SSH 地址或 HTTPS 地址，假设地址为 git@github.com:iissnan/hexo-theme-fluid.git
4、cd themes
5、git clone git@github.com:iissnan/hexo-theme-fluid.git
6、cd ..
7、将 _config.yml 的第 102 行改为 theme: hexo-theme-fluid，保存
8、hexo generate
9、hexo deploy
10、等一分钟，然后刷新你的博客页面，你会看到一个新的外观。如果不喜欢这个主题，就回到第 1 步，重选一个主题。
```



### 4 上传博客源码

「你的用户名.github.io」上保存的只是你的博客，并没有保存「生成博客的程序代码」，你需要再创建一个名为 blog-generator 的空仓库，用来保存 myBlog 里面的「生成博客的程序代码」

```java
1、在 GitHub 创建 blog-generator 空仓库。
2、博客发布在了「你的用户名.github.io」而你的「生成博客的程序代码」发布在了 blog-generator。
3、每次 hexo deploy 完之后，博客就会更新；然后你还要要 add / commit /push 一下「生成博客的程序代码」，以防万一。这个 blog-generator 就是用来生成博客的程序，而「你的用户名.github.io」仓库就是你的博客页面。
4、由于themes中可能含有.git文件，所以add和commit之前先rm -rf .git
```



### 5 一些问题

1、出现显示显示问题bug:  npm i hexo-renderer-swig

2、添加标签和分类

hexo new page "category" 和hexo new page "tag"，将会在source中出现同名文件夹/index.md

在index.md的头添加 `type: "categories"`or `type: "tags"`

在_posts文件中文章.md的头添加

```
tags:
- blog
- Java  # 注意，标签之间是并列

categories:
- Fu 
- Zi  # 注意，分类是父子关系

除非
categories:
- [A]
- [B]
```

3、本地搜索的开启（安装插件）

```java
npm install hexo-generator-search --save

npm install hexo-generator-searchdb --save
```

4、关于一些修改路径：主题的图片更换是在themes的img中，主要修改站点的\_config.yml和themes的\_config.yml

5、分页设置

安装插件

```java
npm install hexo-generator-index --save
npm install hexo-generator-archive – save
npm install hexo-generator-tag --save
npm install hexo-generator-category --save
npm install hexo-generator-about --save
```

\_config.yml的设置

```
index_generator:
  path: ''
  per_page: 6
  order_by: -date
```

6、首页显示摘要

正文以上加上 <!-- more -->

\_config.yml的设置

```
auto_excerpt:
  enable: true
```

