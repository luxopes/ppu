// Spolecne aritmeticke funkce BitTF (= sw/bittf/isa/bt_isa.c): rsr, srd, bitlen, pow2q, fmix32.

// posun vlevo, mnozstvi orezane na 63 (jako simulator ISA)
function signed [63:0] f_shl(input signed [63:0] v, input signed [63:0] s);
    begin
        f_shl = v <<< ((s > 63) ? 6'd63 : s[5:0]);
    end
endfunction

// v / 2^s zaokrouhlene pul nahoru; s <= 0: posun vlevo
function signed [63:0] f_rsr(input signed [63:0] v, input signed [63:0] s);
    reg [5:0] sp;
    reg signed [63:0] h;
    begin
        if (s > 0) begin
            sp = (s > 62) ? 6'd62 : s[5:0];
            h = (64'sd1 <<< sp) >>> 1;
            f_rsr = (v + h) >>> sp;
        end else
            f_rsr = f_shl(v, -s);
    end
endfunction

// stochasticke zaokrouhleni: (v + (r & (2^s - 1))) >> s, s <= 32
function signed [63:0] f_srd(input signed [63:0] v, input signed [63:0] s, input [31:0] r);
    reg [5:0] sp;
    reg signed [63:0] m;
    begin
        if (s > 0) begin
            sp = (s > 32) ? 6'd32 : s[5:0];
            m = (64'sd1 <<< sp) - 64'sd1;
            f_srd = (v + ($signed({32'd0, r}) & m)) >>> sp;
        end else
            f_srd = f_shl(v, -s);
    end
endfunction

function [6:0] f_bitlen(input [63:0] x);
    integer k;
    begin
        f_bitlen = 7'd0;
        for (k = 0; k < 64; k = k + 1)
            if (x[k]) f_bitlen = k[6:0] + 7'd1;
    end
endfunction

// NRMS: s = bitlen(v) - 16, zaokrouhleno nahoru na sude
function signed [63:0] f_nrms(input signed [63:0] v);
    reg signed [63:0] s;
    begin
        s = $signed({57'd0, f_bitlen(v)}) - 64'sd16;
        f_nrms = s + (s & 64'sd1);
    end
endfunction

// gradient pro dW: znamenko * 2^k nad 2^lo_e, nestranne
function signed [63:0] f_pow2q(input signed [63:0] m, input [6:0] lo_e, input [31:0] r);
    reg signed [63:0] a, b, lo, v, rr;
    reg [6:0] bl;
    begin
        a = m[63] ? -m : m;
        bl = f_bitlen(a);
        b = (bl == 0) ? 64'sd1 : (64'sd1 <<< (bl - 7'd1));
        lo = 64'sd1 <<< lo_e[5:0];
        rr = $signed({32'd0, r});
        if (a >= lo)
            v = ((rr & (b - 64'sd1)) < a - b) ? (b <<< 1) : b;
        else
            v = ((rr & (lo - 64'sd1)) < a) ? lo : 64'sd0;
        v = v >>> lo_e[5:0];
        f_pow2q = m[63] ? -v : (m == 0 ? 64'sd0 : v);
    end
endfunction

function [31:0] f_fmix32(input [31:0] h0);
    reg [31:0] h;
    begin
        h = h0 ^ (h0 >> 16);
        h = h * 32'h85EBCA6B;
        h = h ^ (h >> 13);
        h = h * 32'hC2B2AE35;
        f_fmix32 = h ^ (h >> 16);
    end
endfunction
