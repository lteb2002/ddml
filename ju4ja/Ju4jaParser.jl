

##This module is used to parse the messages from Java for Julia

#parse all params from java format to Julia format
#检查传入的所有参数，转换为Julia格式
function parseAllParams(arrs::Array)
   narrs=[(
   if if2DNumericArray(arr)#如果是二维数组，转换为矩阵
      parse2DNumericArrayAsMatrix(arr)
   elseif ifNumericArray(arr)#如果是数组，转换为向量
      parseNumericArrayAsFloatArray(arr)
   else
      arr
   end
   ) for arr in arrs]
   re=narrs
   # println(re)
   return re
end


##将二维数组转换为矩阵
function parse2DNumericArrayAsMatrix(arrs::Array)
   res = Float32.((transpose(hcat(arrs...))))
   # println(res)
   return res
end

function parseNumericArrayAsFloatArray(arr::Array)
   aa = [x for x in arr]
   aa = Float32.(aa)
   # println(aa)
   return aa
end

#测试是否是一个数组的数组
function ifArrayOfArray(arrs::Array)
   if arrs isa Array
      for e in arrs
         if ! (e isa Array)
            return false
         end
      end
      return true
      # return all(e isa Array for e in arrs)
   else
      return false
   end
end

#测试是否是一个元素类型为数值的一维或二维数组
function ifNumericArray(arrs)
   if arrs isa Array
      ass = vcat(arrs...)
      for e in ass
         if !(e  isa Number)
            return false
         end
      end
      return true
      # return all(e isa Number for e in vcat(arrs...))
   else
      return false
   end
end

#测试是否是一个二维的数值数组
function if2DNumericArray(arrs)
   if arrs isa Array
      return ifArrayOfArray(arrs) && ifNumericArray(arrs)
   else
      return false
   end
end

# begin
#    args=[Any[-3.0, -1.0, -2.0], Any[Any[1.0, 1.0, 3.0], Any[2.0, 2.0, 5.0], Any[4.0, 1.0, 2.0]], Any[30.0, 24.0, 36.0]]
#    ls=parseAllParams(args)
#    println(ls)
#
# end
