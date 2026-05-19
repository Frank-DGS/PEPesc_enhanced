# 在Linux内核中启用IPv4数据包转发功能，让这台Linux机器像路由器一样工作，
# 能够将收到的数据包转发到其他网络接口或主机。没有这个设置，Linux只会处理
# 目标地址是自己的数据包，其它包会被丢弃。
# sysctl：用于在运行时查看和修改内核参数的工具。
# -w：表示写入（write）参数值。
# net.ipv4.ip_forward：这是内核参数，控制IPv4协议栈是否转发数据包。
# 0：禁用IP转发（默认）
# 1：启用IP转发
sysctl -w net.ipv4.ip_forward=1 
# iptables：用户空间工具，用于配置 Linux 内核防火墙（Netfilter）规则。
# -t mangle：指定操作表（table），mangle 表用于修改数据包的元数据（如
# TTL、TOS、Mark），不影响最终目标地址。若不指定，默认是 filter 表。
# -A PREROUTING：指定操作链（chain）和动作，-A = --append（追加规则）。
# PREROUTING 链：数据包进入网络栈后、路由决策之前的钩子点（hook）。也就是在路由之前追加动作
# -p tcp：仅匹配TCP协议
# --source 10.0.0.1：匹配源 IP 地址为 10.0.0.1（可接掩码如 /32）。
# --destination 10.0.2.4：匹配目标 IP 地址为 10.0.2.4。
# -j TPROXY：-j = --jump，跳转到 TPROXY 目标。
# TPROXY 是内核模块，用于透明代理，需要内核编译时开启 CONFIG_NETFILTER_TPROXY。
# --on-port 9999：将匹配的数据包重定向到本机（localhost）的 9999 端口。
# -tproxy-mark 1：给数据包打上防火墙标记（firewall mark），值为 1。此标记用于后续 ip rule 的策略路由匹配。
iptables -t mangle -A PREROUTING -p tcp --source 10.0.0.1 --destination 10.0.2.4 -j TPROXY --on-port 9999 --tproxy-mark 1
# ip：iproute2 工具集的核心命令，替代传统的 ifconfig 和 route。
# rule：操作路由策略数据库（Routing Policy Database, RPDB）。决定数据包使用哪个路由表（table）进行查询。
# 默认有 local、main、default 三张表，优先级递减
# add：添加一条新规则（优先级自动分配，数值越小越优先）。
# fwmark 1：匹配条件，匹配数据包的防火墙标记为 1。
# lookup 101：执行动作，lookup = table，表示查询路由表 101。
ip rule add fwmark 1 lookup 101
# ip route：操作内核路由表（FIB, Forwarding Information Base）。
# add：添加路由条目。
# local：路由类型（type），表示目标地址是本机地址，0.0.0.0/0 设置所有目标地址都当做本地地址接收
# dev：指定数据包从哪个网络接口发出。lo：回环接口，是一个虚拟网卡，功能上等价于"本机→本机"
# 再次收到数据包后，会当做本机数据进行接收
# table 101：这条路由添加到表101（不是默认的main表）
ip route add local 0.0.0.0/0 dev lo table 101



# sysctl -w net.ipv4.ip_forward=1 
# iptables -t mangle -A PREROUTING -p tcp --source 10.0.0.1 --destination 10.0.2.4 -j TPROXY --on-port 9999 --tproxy-mark 1
# ip rule add fwmark 1 lookup 101
# ip route add local 0.0.0.0/0 dev lo table 101