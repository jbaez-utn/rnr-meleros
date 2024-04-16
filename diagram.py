from diagrams import Cluster, Diagram
from diagrams.programming.flowchart import StartEnd, Action, Decision, Database

with Diagram("Test diagram", show=True, outformat="svg"):
    start = StartEnd("Start")
    end = StartEnd("End")

    with Cluster("Services"):
        svc_group = [Action("action1"),
                     Action("web2"),
                     Action("web3")]

    with Cluster("dbs"):
        db_primary = Database("userdb")
        db_primary - [Database("userdb ro")]
        
    start >> svc_group >> db_primary >> end