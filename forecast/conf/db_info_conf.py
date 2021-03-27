## Tibero DB

drv = "com.tmax.tibero.jdbc.TbDriver"
jar = "/app/anaconda3/extlib/tibero6-jdbc.jar"
ip = "102.168.0.54"
port = "54321"
sid = "tbsid"
id = "dw"
pwd = "password"

db_conf = {
    "DRV": drv,
    "JAR": jar,
    "IP": ip,
    "PORT": port,
    "SID": sid,
    "ID": id,
    "PWD": pwd,
    "URL": f"jdbc:tibero:thin:@{ip}:{port}:{sid}"
}