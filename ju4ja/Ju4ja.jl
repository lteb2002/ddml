
#This module works as a server for the calls from Java

module Ju4ja

using Sockets
using JSON

export startServer, closeServer, JavaCallInst, JavaCallResult

_server = Nothing
include("Ju4jaParser.jl")

#
function startServer(;
    port::Int = 6996,
    funcProcessor::Function = executeFunction,
    scriptProcessor::Function = executeScript,
)
    _server = Sockets.listen(port)
    while true
        conn = accept(_server)
        @async begin
            try #执行一次客户端的请求过程，并返回结果
                line = readline(conn)
                #println(line)
                println("数据接收完成！")
                exeResult = parseAndExecute(line, funcProcessor, scriptProcessor)
                write(conn, JSON.json(exeResult))
            catch ex
                println("Connection ended with error $ex")
            finally
                close(conn)
            end
        end
    end
end

#解析调用中的参数并执行相应的操作，返回结果
function parseAndExecute(
    line::String,
    funcProcessor::Function = executeFunction,
    scriptProcessor::Function = executeScript,
)
    result = JavaCallResult("", "", "", "")
    jcall = parseLine(line)

    if jcall.operation == "FUNCTION"
        func = jcall.func
        if (func != "")
            try
                result.id = jcall.id#结果ID设置为请求ID
                args = jcall.args #将参数试图解析为支持的类型
                res = funcProcessor(func, args)
                result.resultStr = res
                result.resultType = typeof(res)
                result.status = "success"
            catch ex
                println(ex)
                result.resultStr = ""
                result.status = "error"
                throw(ex)
            end
        end
    elseif jcall.operation == "SCRIPT"
        script = jcall.script
        #执行脚本
        try
            if (script != null)
                scriptProcessor(script)
                result.status = "success"
            else
                result.status = "null"
            end
        catch ex
            #println(ex)
            result.status = "error"
        end
        result.resultStr = ""
    end
    return result
end

function executeFunction(func::String, args::Array{Any})
    nargs = parseAllParams(args)
    # println("parsed params:",nargs)
    ass = Tuple(nargs)
    # println("Prepare to invoke function $func  with args: $ass")
    result = getfield(Main, Symbol(func))(ass...)
    return result
end

function executeScript(script::String)
    include(script)
end

function parseLine(ln::String)
    dict = JSON.parse(ln)
    filePath = get(dict, "tempFilePath", "")
    println("Temp file path:", filePath)
    # txt = readlines(filePath)
    dict = JSON.parsefile(filePath)
    jcall = JavaCallInst("", "", "", "", "", "", "")
    jcall.operation = get(dict, "operation", "")
    jcall.script = get(dict, "script", "")
    jcall.modn = get(dict, "modn", "")
    jcall.func = get(dict, "func", "")
    jcall.args = get(dict, "args", Any[])
    jcall.resultType = get(dict, "resultType", "")
    if jcall.modn == ""
        jcall.modn = "Main"
    end
    println("JSON parsing finished successfuly.")
    return jcall
end


#关闭Server
function closeServer()
    try
        if _server != Nothing
            close(_server)
        end
    catch ex
        print(ex)
    end
end

#对Java调用的参数解析
mutable struct JavaCallInst
    id::Any#id
    operation::Any#操作类型
    script::Any#脚本名
    modn::Any#模块名
    func::Any#调用的函数名
    args::Any#提供的参数
    resultType::Any#返回值类型

end

#对Java调用的结果返回
mutable struct JavaCallResult
    id::Any#
    resultStr::Any#结果（返回值）
    resultType::Any
    status::Any#函数执行结果状态
end

end
