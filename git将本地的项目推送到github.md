# 一、配置本地Git终端，利用SSH连接Github

- 第一步: 创建本地SSH Key:
```
ssh-keygen -t rsa -C "xxx@xx.com"  // 全部使用默认的，一路回车即可。将xxxx@xx.com邮箱修改为GitHub上面的注册邮箱。
```

- 第二步：成功之后会在~/下生成.ssh文件夹，进去之后打开id_rsa.pub，复制里面的Key。

- 第三步：登录GitHub官网，在Account Settings(账户配置)标签下，选择SSH and GPG keys选项下面的SSH keys，然后点击Add SSH Key。SSH Key的Title随便填，粘贴本地电脑上生成的Key。

- 第四步：测试是否成功：在Git 终端中输入命令：
```
ssh -T git@github.com
```
如果是第一次的会提示是否continue，输入yes就会看到：You've successfully authenticated, but GitHub does not provide shell access 。这就表示已成功连上github。

# 二、将新建工程代码文件提交到本地仓库中
- 第一步：新建一个本地仓库（Repository）文件夹；

- 第二步：在Git终端中进入到创建的本地仓库"根"目录下；

- 第三步：将需要Push到Github上的代码文件以及其他所有文件均拷贝到新建的本地仓库"根"目录下；

- 第四步：在Git终端下使用命令：
```
git init   // 生成“本地仓库”
```

- 第五步：在Git终端下使用命令：
```
git add .      // 将所有文件添加到缓存区中
git status -s  // 查看文件添加状态
```

- 第六步：在Git终端下使用命令：
```
git commit -m "xxxxx(描述)" // 将提交到缓存区的所有文件添加到本地仓库中
```

# 三、在Github官网中点击New repository创建远程仓库

- 第一步：填入Repository name之后，点击Create repository即可(最好不要勾选上初始化README.md后面命令有些影响)。

- 第二步：将完成之后页面中所显示的SSH连接地址拷贝。如：git@github.com:Github用户名/仓库名.git

# 四、将本地仓库传到GitHub上去

- 第一步：设置username和email，因为GitHub每次commit都会记录它们：
```
-------设置命令：git config --global user.name "your name"
-------设置命令：git config --global user.email "your_email@youremail.com"
```

- 第二步：进入要上传的仓库，添加远程地址：
```
-------设置命令：git remote add origin SSH连接地址（如以上拷贝的：git@github.com:Github用户名/仓库名.git）
```

- 第三步：将本地仓库上传到GitHub上
```
-------使用命令：git push -u origin master
```



