# iRRAM踩坑
- 官方2013release版对mac os不友好，无法编译，所以直接在git上down下来，quick install的默认路径也是需要加上的否则无法通过配置
- 默认安装一次除非卸载否则不支持再一次安装
- iRRAM.lib含有main方法的实现，除非修改库文件或者直接使用提供的make方法编译
- 编译部分方法出bug，集中在e^x那一块，修改部分实现成功