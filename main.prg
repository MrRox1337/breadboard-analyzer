Function main
    Motor On
    Power High

    ' Open TCP/IP Port to listen to Python Python Model
    ' #201 is the port identifier from System Configuration
    ' Make sure this matches your defined port settings in RC+
    SetNet #201, "127.0.0.1", 2000, CRLF, NONE, 0
    OpenNet #201 As Server
    
    Print "Waiting for Python Model to connect..."
    WaitNet #201
    Print "Connected to Python Model."

    String result$

    Do
        ' Wait and receive string from Python
        Print "Waiting for classification result..."
        Input #201, result$
        Print "Received: ", result$

        If result$ = "PASS" Then
            Print "Circuit is PASS. Moving to Zone A."
            PerformPickAndPlace(P2) ' Zone A
        ElseIf result$ = "FAIL" Then
            Print "Circuit is FAIL. Moving to Zone B."
            PerformPickAndPlace(P3) ' Zone B
        Else
            Print "Unknown command ignored."
        EndIf
    Loop
Fend

Function PerformPickAndPlace(dropPoint As Point)
    Print "Moving to Pick Location..."
    ' Jump to above the pick location (Z offset by 50mm)
    Jump P1 +Z(50) 
    
    ' Move linearly down to the pick location
    Move P1
    
    ' Activate End Effector (Gripper/Vacuum)
    On 1 
    Wait 0.5
    
    ' Move linearly up
    Move P1 +Z(50)

    Print "Moving to Drop Location..."
    ' Jump to above the drop location (Zone A or Zone B)
    Jump dropPoint +Z(50)
    
    ' Move linearly down to drop position
    Move dropPoint
    
    ' Deactivate End Effector (Gripper/Vacuum)
    Off 1
    Wait 0.5
    
    ' Move linearly up
    Move dropPoint +Z(50)

    Print "Returning Home..."
    ' Return Home
    Jump P0
Fend
