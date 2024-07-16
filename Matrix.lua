--M(i, j) gives the i, j element

-- A * B
-- A + B
-- A - B
-- if A or B is a number, converts to a matrix of compatible dimension with the number as the diagonals

-- A / B
-- A or B must be a number

-- M.I    gives the inverse (up to 4x4)
-- M.T    gives the transpose
-- M.tr   gives the trace
-- M.det  gives the determinant (up to 4x4)
-- M.norm gives the frobenius norm (magnitude) of the matrix
-- M.unit gives M/M.norm

-- M.crossMatrix gives matrix C such that C*N = M:cross(N)

-- A.dot(B)
-- A.cross(B)


local Mat = {}

function Mat.new(h, w, ...)
	return setmetatable({h = h; w = w; ...}, Mat)
end

function Mat.col(...)
	return Mat.new(select("#", ...), 1, ...)
end

function Mat.row(...)
	return Mat.new(1, select("#", ...), ...)
end

function Mat.fromCol(...)
	local h = #(...) -- eh whatever.
	local w = select("#", ...)
	local M = Mat.new(h, w)
	for row = 1, h do
		for col = 1, w do
			local v = select(col, ...)
			M[w*(row - 1) + col] = v[row]
		end
	end

	return M
end

function Mat.null(h, w)
	local M = setmetatable({h = h; w = w;}, Mat)
	for i = 1, h*w do
		M[i] = 0
	end
	return M
end

function Mat.identity(s)
	local M = setmetatable({h = s; w = s;}, Mat)
	for i = 1, s*s do
		if i%(s + 1) == 1 then
			M[i] = 1
		else
			M[i] = 0
		end
	end
	return M
end

function Mat.constructor(h, w)
	return function(...)
		return Mat.new(h, w, ...)
	end
end

function Mat.fromAngle(a)
	local co = math.cos(a)
	local si = math.sin(a)
	return Mat.new(2, 2,
		co, -si,
		si,  co)
end

function Mat.fromRotationVector(r)
	local a = r.norm
	if a < 1e-8 then
		return Mat.identity(3)
	end
	
	local v = r.unit
	local V = v.crossMatrix
	local co = math.cos(a)
	local si = math.sin(a)
	return v*v.T + co*V*V.T + si*V
end

-- creates a scaling rotation matrix which transforms u into v
-- for a pure rotation matrix, give u and v as unit vectors
function Mat.rotate(u, v)
	-- NOTE: there is a more stable way of doing this which relies on inner product
	-- but complexity squares with the dimension of u and v
	local uu = u:dot(u)
	local uv = u:dot(v)
	local vv = v:dot(v)
	local w = 2*uv/uu*v - vv/uu*u -- geometric algebra: v*u^-1*v

	local A = Mat.fromCol(u, v)
	local B = Mat.fromCol(v, w)
	local T = (A.T*A).I*A.T -- T*p gives t such that A*t = (p projected onto plane u, v)

	return 1 + (B - A)*T
end

function Mat:copy()
	return Mat.new(self.h, self.w, table.unpack(self))
end

function Mat:__call(row, col, ...)
	return self[self.w*(row - 1) + col]
end

-- returns the submatrix
function Mat.sub(A, i, j, h, w)
	local S = Mat.new(h, w)
	for row = 1, h do
		for col = 1, w do
			S[w*(row - 1) + col] = A[A.w*(i + row - 2) + j + col - 1]
		end
	end
end

-- right now only defined in 3D
function Mat.cross(A, B)
	if A.h*A.w ~= 3 or B.h*B.w ~= 3 then
		error(`{A.h}x{A.w} matrix cross {B.h}x{B.w} matrix is undefined`, 0)
	end
	
	return Mat.col(A[2]*B[3] - A[3]*B[2], A[3]*B[1] - A[1]*B[3], A[1]*B[2] - A[2]*B[1])
end

function Mat.dot(A, B)
	if A.w ~= B.w or A.h ~= B.h then
		error(`{A.h}x{A.w} matrix dot {B.h}x{B.w} matrix is undefined`, 0)
	end
	
	local h, w = A.h, A.w
	local sum = 0
	for i = 1, h*w do
		sum = sum + A[i]*B[i]
	end
	return sum
end

function Mat:__index(index)
	if Mat[index] then
		return Mat[index]
	end

	local w, h = self.w, self.h
	if index == "T" then
		local T = Mat.new(w, h)
		for row = 1, h do
			for col = 1, w do
				T[h*(col - 1) + row] = self[w*(row - 1) + col]
			end
		end
		return T
	elseif index == "tr" then
		local sum = 0
		for i = 1, math.min(w, h) do
			sum = sum + self[w*(i - 1) + i]
		end
		return sum
	elseif index == "I" then
		if w ~= h then
			error(`({w}x{h} matrix).I is undefined; matrix must be square`, 0)
		end

		if w == 1 then
			return Mat.new(1, 1, 1/self[1])
		elseif w == 2 then
			local xx, yx = self[1], self[2]
			local xy, yy = self[3], self[4]
			local det = xx*yy - xy*yx
			return Mat.new(2, 2,
				yy/det, -yx/det,
				-xy/det,  xx/det)
		elseif w == 3 then
			local xx, yx, zx = self[1], self[2], self[3]
			local xy, yy, zy = self[4], self[5], self[6]
			local xz, yz, zz = self[7], self[8], self[9]
			local det = xx*(yy*zz - yz*zy) + xy*(yz*zx - yx*zz) + xz*(yx*zy - yy*zx)
			return Mat.new(3, 3,
				(yy*zz - yz*zy)/det, (yz*zx - yx*zz)/det, (yx*zy - yy*zx)/det,
				(xz*zy - xy*zz)/det, (xx*zz - xz*zx)/det, (xy*zx - xx*zy)/det,
				(xy*yz - xz*yy)/det, (xz*yx - xx*yz)/det, (xx*yy - xy*yx)/det)
		elseif w == 4 then
			local ww, xw, yw, zw = self[01], self[02], self[03], self[04]
			local wx, xx, yx, zx = self[05], self[06], self[07], self[08]
			local wy, xy, yy, zy = self[09], self[10], self[11], self[12]
			local wz, xz, yz, zz = self[13], self[14], self[15], self[16]
			local det =
				(ww*xx - wx*xw)*(yy*zz - yz*zy) - (ww*xy - wy*xw)*(yx*zz - yz*zx) + 
				(ww*xz - wz*xw)*(yx*zy - yy*zx) + (wx*xy - wy*xx)*(yw*zz - yz*zw) - 
				(wx*xz - wz*xx)*(yw*zy - yy*zw) + (wy*xz - wz*xy)*(yw*zx - yx*zw)
			return Mat.new(4, 4,
				(xx*(yy*zz - yz*zy) + xy*(yz*zx - yx*zz) + xz*(yx*zy - yy*zx))/det,
				(xw*(yz*zy - yy*zz) + xy*(yw*zz - yz*zw) + xz*(yy*zw - yw*zy))/det,
				(xw*(yx*zz - yz*zx) + xx*(yz*zw - yw*zz) + xz*(yw*zx - yx*zw))/det,
				(xw*(yy*zx - yx*zy) + xx*(yw*zy - yy*zw) + xy*(yx*zw - yw*zx))/det,
				(wx*(yz*zy - yy*zz) + wy*(yx*zz - yz*zx) + wz*(yy*zx - yx*zy))/det,
				(ww*(yy*zz - yz*zy) + wy*(yz*zw - yw*zz) + wz*(yw*zy - yy*zw))/det,
				(ww*(yz*zx - yx*zz) + wx*(yw*zz - yz*zw) + wz*(yx*zw - yw*zx))/det,
				(ww*(yx*zy - yy*zx) + wx*(yy*zw - yw*zy) + wy*(yw*zx - yx*zw))/det,
				(wx*(xy*zz - xz*zy) + wy*(xz*zx - xx*zz) + wz*(xx*zy - xy*zx))/det,
				(ww*(xz*zy - xy*zz) + wy*(xw*zz - xz*zw) + wz*(xy*zw - xw*zy))/det,
				(ww*(xx*zz - xz*zx) + wx*(xz*zw - xw*zz) + wz*(xw*zx - xx*zw))/det,
				(ww*(xy*zx - xx*zy) + wx*(xw*zy - xy*zw) + wy*(xx*zw - xw*zx))/det,
				(wx*(xz*yy - xy*yz) + wy*(xx*yz - xz*yx) + wz*(xy*yx - xx*yy))/det,
				(ww*(xy*yz - xz*yy) + wy*(xz*yw - xw*yz) + wz*(xw*yy - xy*yw))/det,
				(ww*(xz*yx - xx*yz) + wx*(xw*yz - xz*yw) + wz*(xx*yw - xw*yx))/det,
				(ww*(xx*yy - xy*yx) + wx*(xy*yw - xw*yy) + wy*(xw*yx - xx*yw))/det)
		else
			-- do some gauss jordan elimination here in the future
			error("inverse not yet supported for dimensions higher than 4", 0)
		end
	elseif index == "det" then
		if w ~= h then
			error(`({w}x{h} matrix).det is undefined; matrix must be square`, 0)
		end
		
		if w == 1 then
			return self[1]
		elseif w == 2 then
			local xx, yx = self[1], self[2]
			local xy, yy = self[3], self[4]
			return xx*yy - xy*yx
		elseif w == 3 then
			local xx, yx, zx = self[1], self[2], self[3]
			local xy, yy, zy = self[4], self[5], self[6]
			local xz, yz, zz = self[7], self[8], self[9]
			return xx*(yy*zz - yz*zy) + xy*(yz*zx - yx*zz) + xz*(yx*zy - yy*zx)
		elseif w == 4 then
			local ww, xw, yw, zw = self[01], self[02], self[03], self[04]
			local wx, xx, yx, zx = self[05], self[06], self[07], self[08]
			local wy, xy, yy, zy = self[09], self[10], self[11], self[12]
			local wz, xz, yz, zz = self[13], self[14], self[15], self[16]
			return -- the 4x4 determinant has this really pretty factorization
				(ww*xx - wx*xw)*(yy*zz - yz*zy) - (ww*xy - wy*xw)*(yx*zz - yz*zx) + 
				(ww*xz - wz*xw)*(yx*zy - yy*zx) + (wx*xy - wy*xx)*(yw*zz - yz*zw) - 
				(wx*xz - wz*xx)*(yw*zy - yy*zw) + (wy*xz - wz*xy)*(yw*zx - yx*zw)
		else
			-- fully expanded, determinants take n! (n - 1) multiplies
			-- recursive cofactor expansion, determinants take floor((e - 1) x! - 1) multiplies (!?)
			-- with a wedge product expansion, determinants take 2^n n/2 - n multiplies
			-- using gauss jordan elimination, determinants take (2 n - 1) (n - 1) n/6 multiplies and (n - 1) n/2 divides
			error("determinant not yet supported for dimensions higher than 4", 0)
		end
	elseif index == "norm" then
		local sum = 0
		for i = 1, h*w do
			sum = sum + self[i]*self[i]
		end
		return math.sqrt(sum)
	elseif index == "unit" then
		local norm = self.norm
		if norm == 0 then
			return Mat.null(h, w)
		end
		local U = Mat.new(h, w)
		for i = 1, h*w do
			U[i] = self[i]/norm
		end
		return U
	elseif index == "crossMatrix" then
		-- we can define cross matrix for two 4d vectors, too, in the future
		if w ~= 1 and h ~= 1 and w*h ~= 3 then
			error(`({h}x{w} matrix).crossMatrix is undefined; matrix must be 3x1 or 1x3`, 0)
		end
		
		return Mat.new(3, 3,
			       0, -self[3],  self[2],
			 self[3],        0, -self[1],
			-self[2],  self[1],        0)
	end
end

function Mat.__add(A, B)
	local typeA = getmetatable(A) == Mat and "matrix" or type(A)
	local typeB = getmetatable(B) == Mat and "matrix" or type(B)
	if typeA == "number" and typeB == "matrix" then
		local C = Mat.new(B.h, B.w)
		for i = 1, B.h*B.w do
			C[i] = B[i]
		end
		for i = 1, B.h*B.w, B.w + 1 do
			C[i] = A + C[i]
		end
		return C
	elseif typeA == "matrix" and typeB == "number" then
		local C = Mat.new(A.h, A.w)
		for i = 1, A.h*A.w do
			C[i] = A[i]
		end
		for i = 1, A.h*A.w, A.w + 1 do
			C[i] = C[i] + B
		end
		return C
	elseif typeA == "matrix" and typeB == "matrix" then
		if A.w ~= B.w or A.h ~= B.h then
			error(`{A.h}x{A.w} matrix + {B.h}x{B.w} matrix is undefined`, 0)
		end
		
		local C = Mat.new(A.h, A.w)
		for i = 1, A.h*A.w do
			C[i] = A[i] + B[i]
		end

		return C
	end
	
	error(`{typeA} + {typeB} is undefined`, 0)
end

function Mat.__sub(A, B)
	local typeA = getmetatable(A) == Mat and "matrix" or type(A)
	local typeB = getmetatable(B) == Mat and "matrix" or type(B)
	if typeA == "number" and typeB == "matrix" then
		local C = Mat.new(B.h, B.w)
		for i = 1, B.h*B.w do
			C[i] = -B[i]
		end
		for i = 1, B.h*B.w, B.w + 1 do
			C[i] = A + C[i]
		end
		return C
	elseif typeA == "matrix" and typeB == "number" then
		local C = Mat.new(A.h, A.w)
		for i = 1, A.h*A.w do
			C[i] = A[i]
		end
		for i = 1, A.h*A.w, A.w + 1 do
			C[i] = C[i] - B
		end
		return C
	elseif typeA == "matrix" and typeB == "matrix" then
		if A.w ~= B.w or A.h ~= B.h then
			error(`{A.h}x{A.w} matrix - {B.h}x{B.w} matrix is undefined`, 0)
		end
		
		local C = Mat.new(A.h, A.w)
		for i = 1, A.h*A.w do
			C[i] = A[i] - B[i]
		end

		return C
	end

	error(`{typeA} - {typeB} is undefined`, 0)
end

function Mat.__unm(A)
	local B = Mat.new(A.h, A.w)
	for i = 1, A.h*A.w do
		B[i] = -A[i]
	end

	return B
end

function Mat.__mul(A, B)
	local typeA = getmetatable(A) == Mat and "matrix" or type(A)
	local typeB = getmetatable(B) == Mat and "matrix" or type(B)
	if typeA == "number" and typeB == "matrix" then
		local C = Mat.new(B.h, B.w)
		for i = 1, B.h*B.w do
			C[i] = A*B[i]
		end
		return C
	elseif typeA == "matrix" and typeB == "number" then
		local C = Mat.new(A.h, A.w)
		for i = 1, A.h*A.w do
			C[i] = A[i]*B
		end
		return C
	elseif typeA == "matrix" and typeB == "matrix" then
		if A.w ~= B.h then
			error(`{A.h}x{A.w} matrix * {B.h}x{B.w} matrix is undefined`, 0)
		end
		
		local C = Mat.new(A.h, B.w)
		for row = 1, A.h do
			for col = 1, B.w do
				local sum = 0
				for i = 1, A.w do
					sum = sum + A[A.w*(row - 1) + i]*B[B.w*(i - 1) + col]
				end
				C[C.w*(row - 1) + col] = sum
			end
		end

		return C
	end

	error(`{typeA} * {typeB} is undefined`, 0)
end

function Mat.__div(A, B)
	local typeA = getmetatable(A) == Mat and "matrix" or type(A)
	local typeB = getmetatable(B) == Mat and "matrix" or type(B)
	if typeA == "number" and typeB == "matrix" then
		return A*B.I
	elseif typeA == "matrix" and typeB == "number" then
		return A*(1/B)
	elseif typeA == "matrix" and typeB == "matrix" then
		error("matrix / matrix is ambiguous; use A*B.I for right division and B.I*A for left division")
	end

	error(`{typeA} / {typeB} is undefined`, 0)
end

local matStr = {}
local rowStr = {}
function Mat.__tostring(A)
	table.clear(matStr)
	table.clear(rowStr)
	local h, w = A.h, A.w
	for row = 1, h do
		for col = 1, w do
			rowStr[col] = string.format("%+.3f", A[w*(row - 1) + col])
		end
		matStr[row] = table.concat(rowStr, ", ")
	end

	return table.concat(matStr, " | ")
end

return Mat
