(function(){'use strict';var p,aa=typeof Object.create=="function"?Object.create:function(a){function b(){}
b.prototype=a;return new b},ba=typeof Object.defineProperties=="function"?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;
a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var da=ca(this);function u(a,b){if(b)a:{var c=da;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&b!=null&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
var ea=function(){function a(){function c(){}
new c;Reflect.construct(c,[],function(){});
return new c instanceof c}
if(typeof Reflect!="undefined"&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(c,d,e){c=b(c,d);e&&Reflect.setPrototypeOf(c,e.prototype);return c}}return function(c,d,e){e===void 0&&(e=c);
e=aa(e.prototype||Object.prototype);return Function.prototype.apply.call(c,e,d)||e}}(),fa;
if(typeof Object.setPrototypeOf=="function")fa=Object.setPrototypeOf;else{var ha;a:{var ia={a:!0},ja={};try{ja.__proto__=ia;ha=ja.a;break a}catch(a){}ha=!1}fa=ha?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var la=fa;
function v(a,b){a.prototype=aa(b.prototype);a.prototype.constructor=a;if(la)la(a,b);else for(var c in b)if(c!="prototype")if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.Ca=b.prototype}
function ma(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
function y(a){var b=typeof Symbol!="undefined"&&Symbol.iterator&&a[Symbol.iterator];if(b)return b.call(a);if(typeof a.length=="number")return{next:ma(a)};throw Error(String(a)+" is not an iterable or ArrayLike");}
function A(a){if(!(a instanceof Array)){a=y(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function pa(a){return qa(a,a)}
function qa(a,b){a.raw=b;Object.freeze&&(Object.freeze(a),Object.freeze(b));return a}
function ra(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var sa=typeof Object.assign=="function"?Object.assign:function(a,b){if(a==null)throw new TypeError("No nullish arg");a=Object(a);for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)ra(d,e)&&(a[e]=d[e])}return a};
u("Object.assign",function(a){return a||sa});
function ta(){this.B=!1;this.o=null;this.i=void 0;this.h=1;this.u=this.H=0;this.K=this.j=null}
function ua(a){if(a.B)throw new TypeError("Generator is already running");a.B=!0}
ta.prototype.G=function(a){this.i=a};
function va(a,b){a.j={exception:b,Yd:!0};a.h=a.H||a.u}
ta.prototype.return=function(a){this.j={return:a};this.h=this.u};
ta.prototype.yield=function(a,b){this.h=b;return{value:a}};
ta.prototype.v=function(a){this.h=a};
function wa(a,b,c){a.H=b;c!=void 0&&(a.u=c)}
function xa(a,b){a.h=b;a.H=0}
function ya(a){a.H=0;var b=a.j.exception;a.j=null;return b}
function za(a){a.K=[a.j];a.H=0;a.u=0}
function Aa(a,b){var c=a.K.splice(0)[0];(c=a.j=a.j||c)?c.Yd?a.h=a.H||a.u:c.v!=void 0&&a.u<c.v?(a.h=c.v,a.j=null):a.h=a.u:a.h=b}
function Ba(a){this.h=new ta;this.i=a}
function Ca(a,b){ua(a.h);var c=a.h.o;if(c)return Da(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return Ea(a)}
function Da(a,b,c,d){try{var e=b.call(a.h.o,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.B=!1,e;var f=e.value}catch(g){return a.h.o=null,va(a.h,g),Ea(a)}a.h.o=null;d.call(a.h,f);return Ea(a)}
function Ea(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.B=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,va(a.h,c)}a.h.B=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.Yd)throw b.exception;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Fa(a){this.next=function(b){ua(a.h);a.h.o?b=Da(a,a.h.o.next,b,a.h.G):(a.h.G(b),b=Ea(a));return b};
this.throw=function(b){ua(a.h);a.h.o?b=Da(a,a.h.o["throw"],b,a.h.G):(va(a.h,b),b=Ea(a));return b};
this.return=function(b){return Ca(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ha(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function B(a){return Ha(new Fa(new Ba(a)))}
function C(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
u("globalThis",function(a){return a||da});
u("Reflect",function(a){return a?a:{}});
u("Reflect.construct",function(){return ea});
u("Reflect.setPrototypeOf",function(a){return a?a:la?function(b,c){try{return la(b,c),!0}catch(d){return!1}}:null});
u("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(Math.random()*1E9>>>0)+"_",e=0;return b});
u("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");ba(Array.prototype,a,{configurable:!0,writable:!0,value:function(){return Ia(ma(this))}});
return a});
function Ia(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
u("Promise",function(a){function b(g){this.Z=0;this.ib=void 0;this.h=[];this.u=!1;var h=this.i();try{g(h.resolve,h.reject)}catch(k){h.reject(k)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(this.h==null){this.h=[];var h=this;this.j(function(){h.u()})}this.h.push(g)};
var e=da.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.u=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var k=g[h];g[h]=null;try{k()}catch(l){this.o(l)}}}this.h=null};
c.prototype.o=function(g){this.j(function(){throw g;})};
b.prototype.i=function(){function g(l){return function(m){k||(k=!0,l.call(h,m))}}
var h=this,k=!1;return{resolve:g(this.V),reject:g(this.j)}};
b.prototype.V=function(g){if(g===this)this.j(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.ba(g);else{a:switch(typeof g){case "object":var h=g!=null;break a;case "function":h=!0;break a;default:h=!1}h?this.K(g):this.o(g)}};
b.prototype.K=function(g){var h=void 0;try{h=g.then}catch(k){this.j(k);return}typeof h=="function"?this.ha(h,g):this.o(g)};
b.prototype.j=function(g){this.H(2,g)};
b.prototype.o=function(g){this.H(1,g)};
b.prototype.H=function(g,h){if(this.Z!=0)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.Z);this.Z=g;this.ib=h;this.Z===2&&this.aa();this.B()};
b.prototype.aa=function(){var g=this;e(function(){if(g.G()){var h=da.console;typeof h!=="undefined"&&h.error(g.ib)}},1)};
b.prototype.G=function(){if(this.u)return!1;var g=da.CustomEvent,h=da.Event,k=da.dispatchEvent;if(typeof k==="undefined")return!0;typeof g==="function"?g=new g("unhandledrejection",{cancelable:!0}):typeof h==="function"?g=new h("unhandledrejection",{cancelable:!0}):(g=da.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.ib;return k(g)};
b.prototype.B=function(){if(this.h!=null){for(var g=0;g<this.h.length;++g)f.i(this.h[g]);this.h=null}};
var f=new c;b.prototype.ba=function(g){var h=this.i();g.Cc(h.resolve,h.reject)};
b.prototype.ha=function(g,h){var k=this.i();try{g.call(h,k.resolve,k.reject)}catch(l){k.reject(l)}};
b.prototype.then=function(g,h){function k(r,t){return typeof r=="function"?function(w){try{l(r(w))}catch(z){m(z)}}:t}
var l,m,n=new b(function(r,t){l=r;m=t});
this.Cc(k(g,l),k(h,m));return n};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.Cc=function(g,h){function k(){switch(l.Z){case 1:g(l.ib);break;case 2:h(l.ib);break;default:throw Error("Unexpected state: "+l.Z);}}
var l=this;this.h==null?f.i(k):this.h.push(k);this.u=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,k){k(g)})};
b.race=function(g){return new b(function(h,k){for(var l=y(g),m=l.next();!m.done;m=l.next())d(m.value).Cc(h,k)})};
b.all=function(g){var h=y(g),k=h.next();return k.done?d([]):new b(function(l,m){function n(w){return function(z){r[w]=z;t--;t==0&&l(r)}}
var r=[],t=0;do r.push(void 0),t++,d(k.value).Cc(n(r.length-1),m),k=h.next();while(!k.done)})};
return b});
u("Object.setPrototypeOf",function(a){return a||la});
u("Symbol.dispose",function(a){return a?a:Symbol("Symbol.dispose")});
u("SuppressedError",function(a){function b(c,d,e){if(!(this instanceof b))return new b(c,d,e);e=Error(e);"stack"in e&&(this.stack=e.stack);this.message=e.message;this.error=c;this.suppressed=d}
if(a)return a;v(b,Error);b.prototype.name="SuppressedError";return b});
u("WeakMap",function(a){function b(k){this.h=(h+=Math.random()+1).toString();if(k){k=y(k);for(var l;!(l=k.next()).done;)l=l.value,this.set(l[0],l[1])}}
function c(){}
function d(k){var l=typeof k;return l==="object"&&k!==null||l==="function"}
function e(k){if(!ra(k,g)){var l=new c;ba(k,g,{value:l})}}
function f(k){var l=Object[k];l&&(Object[k]=function(m){if(m instanceof c)return m;Object.isExtensible(m)&&e(m);return l(m)})}
if(function(){if(!a||!Object.seal)return!1;try{var k=Object.seal({}),l=Object.seal({}),m=new a([[k,2],[l,3]]);if(m.get(k)!=2||m.get(l)!=3)return!1;m.delete(k);m.set(l,4);return!m.has(k)&&m.get(l)==4}catch(n){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(k,l){if(!d(k))throw Error("Invalid WeakMap key");e(k);if(!ra(k,g))throw Error("WeakMap key fail: "+k);k[g][this.h]=l;return this};
b.prototype.get=function(k){return d(k)&&ra(k,g)?k[g][this.h]:void 0};
b.prototype.has=function(k){return d(k)&&ra(k,g)&&ra(k[g],this.h)};
b.prototype.delete=function(k){return d(k)&&ra(k,g)&&ra(k[g],this.h)?delete k[g][this.h]:!1};
return b});
u("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,k){var l=h[1];return Ia(function(){if(l){for(;l.head!=h[1];)l=l.previous;for(;l.next!=l.head;)return l=l.next,{done:!1,value:k(l)};l=null}return{done:!0,value:void 0}})}
function d(h,k){var l=k&&typeof k;l=="object"||l=="function"?f.has(k)?l=f.get(k):(l=""+ ++g,f.set(k,l)):l="p_"+k;var m=h[0][l];if(m&&ra(h[0],l))for(h=0;h<m.length;h++){var n=m[h];if(k!==k&&n.key!==n.key||k===n.key)return{id:l,list:m,index:h,entry:n}}return{id:l,list:m,index:-1,entry:void 0}}
function e(h){this[0]={};this[1]=b();this.size=0;if(h){h=y(h);for(var k;!(k=h.next()).done;)k=k.value,this.set(k[0],k[1])}}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var h=Object.seal({x:4}),k=new a(y([[h,"s"]]));if(k.get(h)!="s"||k.size!=1||k.get({x:4})||k.set({x:4},"t")!=k||k.size!=2)return!1;var l=k.entries(),m=l.next();if(m.done||m.value[0]!=h||m.value[1]!="s")return!1;m=l.next();return m.done||m.value[0].x!=4||m.value[1]!="t"||!l.next().done?!1:!0}catch(n){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,k){h=h===0?0:h;var l=d(this,h);l.list||(l.list=this[0][l.id]=[]);l.entry?l.entry.value=k:(l.entry={next:this[1],previous:this[1].previous,head:this[1],key:h,value:k},l.list.push(l.entry),this[1].previous.next=l.entry,this[1].previous=l.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this[0][h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this[0]={};this[1]=this[1].previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,k){for(var l=this.entries(),m;!(m=l.next()).done;)m=m.value,h.call(k,m[1],m[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
u("Set",function(a){function b(c){this.h=new Map;if(c){c=y(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||typeof a!="function"||!a.prototype.entries||typeof Object.seal!="function")return!1;try{var c=Object.seal({x:4}),d=new a(y([c]));if(!d.has(c)||d.size!=1||d.add(c)!=d||d.size!=1||d.add({x:4})!=d||d.size!=2)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||f.value[0].x!=4||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=c===0?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
function Ja(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
u("Array.prototype.entries",function(a){return a?a:function(){return Ja(this,function(b,c){return[b,c]})}});
u("Array.prototype.keys",function(a){return a?a:function(){return Ja(this,function(b){return b})}});
function Ka(a,b,c){if(a==null)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
u("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
u("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"endsWith");b+="";c===void 0&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;e>0&&c>0;)if(d[--c]!=b[--e])return!1;return e<=0}});
u("Number.isFinite",function(a){return a?a:function(b){return typeof b!=="number"?!1:!isNaN(b)&&b!==Infinity&&b!==-Infinity}});
u("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
u("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)ra(b,d)&&c.push(b[d]);return c}});
u("Object.is",function(a){return a?a:function(b,c){return b===c?b!==0||1/b===1/c:b!==b&&c!==c}});
u("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(c<0&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
u("String.prototype.includes",function(a){return a?a:function(b,c){return Ka(this,b,"includes").indexOf(b,c||0)!==-1}});
u("Array.from",function(a){return a?a:function(b,c,d){c=c!=null?c:function(h){return h};
var e=[],f=typeof Symbol!="undefined"&&Symbol.iterator&&b[Symbol.iterator];if(typeof f=="function"){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
u("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)ra(b,d)&&c.push([d,b[d]]);return c}});
u("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
u("Number.MIN_SAFE_INTEGER",function(){return-9007199254740991});
u("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
u("Number.isSafeInteger",function(a){return a?a:function(b){return Number.isInteger(b)&&Math.abs(b)<=Number.MAX_SAFE_INTEGER}});
u("Math.trunc",function(a){return a?a:function(b){b=Number(b);if(isNaN(b)||b===Infinity||b===-Infinity||b===0)return b;var c=Math.floor(Math.abs(b));return b<0?-c:c}});
u("Number.isNaN",function(a){return a?a:function(b){return typeof b==="number"&&isNaN(b)}});
u("Array.prototype.values",function(a){return a?a:function(){return Ja(this,function(b,c){return c})}});
u("Promise.prototype.finally",function(a){return a?a:function(b){return this.then(function(c){return Promise.resolve(b()).then(function(){return c})},function(c){return Promise.resolve(b()).then(function(){throw c;
})})}});
u("Math.imul",function(a){return a?a:function(b,c){b=Number(b);c=Number(c);var d=b&65535,e=c&65535;return d*e+((b>>>16&65535)*e+d*(c>>>16&65535)<<16>>>0)|0}});/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
var La=La||{},D=this||self;function E(a,b,c){a=a.split(".");c=c||D;for(var d;a.length&&(d=a.shift());)a.length||b===void 0?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function Ma(a,b){var c=G("CLOSURE_FLAGS");a=c&&c[a];return a!=null?a:b}
function G(a,b){a=a.split(".");b=b||D;for(var c=0;c<a.length;c++)if(b=b[a[c]],b==null)return null;return b}
function Na(a){var b=typeof a;return b!="object"?b:a?Array.isArray(a)?"array":b:"null"}
function Oa(a){var b=Na(a);return b=="array"||b=="object"&&typeof a.length=="number"}
function Pa(a){var b=typeof a;return b=="object"&&a!=null||b=="function"}
function Qa(a){return Object.prototype.hasOwnProperty.call(a,Ta)&&a[Ta]||(a[Ta]=++Ua)}
var Ta="closure_uid_"+(Math.random()*1E9>>>0),Ua=0;function Va(a,b,c){return a.call.apply(a.bind,arguments)}
function Wa(a,b,c){if(!a)throw Error();if(arguments.length>2){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Xa(a,b,c){Xa=Function.prototype.bind&&Function.prototype.bind.toString().indexOf("native code")!=-1?Va:Wa;return Xa.apply(null,arguments)}
function Ya(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}}
function Za(){return Date.now()}
function $a(a){return a}
function ab(a,b){function c(){}
c.prototype=b.prototype;a.Ca=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.base=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
;function bb(a){var b=C.apply(1,arguments).filter(Boolean).join("&");if(!b)return a;var c=a.match(/[?&]adurl=/);return c?a.slice(0,c.index+1)+b+"&"+a.slice(c.index+1):a+(a.indexOf("?")<0?"?":"&")+b}
function cb(a,b){return b?"&"+a+"="+encodeURIComponent(b):""}
function db(a){var b=a.url;a=a.Hj;this.i=b;this.o=a;this.j=(new Date).getTime()-17040672E5;this.h={};for(var c=/[?&]([^&=]+)=([^&]*)/g;a=c.exec(b);)this.h[a[1]]=a[2]}
function eb(a){a=a.o;if(!a)return"";var b=cb("uap",a.platform)+cb("uapv",a.platformVersion)+cb("uafv",a.uaFullVersion)+cb("uaa",a.architecture)+cb("uam",a.model)+cb("uab",a.bitness);a.fullVersionList&&(b+="&uafvl="+encodeURIComponent(a.fullVersionList.map(function(c){return encodeURIComponent(c.brand)+";"+encodeURIComponent(c.version)}).join("|")));
a.wow64!=null&&(b+="&uaw="+Number(a.wow64));return b.slice(1)}
;function fb(a,b){if(b!==null&&b!==void 0){if(typeof b!=="object"&&typeof b!=="function")throw new TypeError("Object expected.");if(c===void 0){if(!Symbol.dispose)throw new TypeError("Symbol.dispose is not defined.");var c=b[Symbol.dispose]}if(typeof c!=="function")throw new TypeError("Object not disposable.");a.stack.push({value:b,dispose:c,async:!1})}return b}
function gb(a){function b(f){a.error=a.ob?new SuppressedError(f,a.error,"An error was suppressed during disposal."):f;a.ob=!0}
function c(){for(;d=a.stack.pop();)try{if(!d.async&&e===1)return e=0,a.stack.push(d),Promise.resolve().then(c);if(d.dispose){var f=d.dispose.call(d.value);if(d.async)return e|=2,Promise.resolve(f).then(c,function(g){b(g);return c()})}else e|=1}catch(g){b(g)}if(e===1)return a.ob?Promise.reject(a.error):Promise.resolve();
if(a.ob)throw a.error;}
var d,e=0;c()}
;function ib(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,ib);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));b!==void 0&&(this.cause=b)}
ab(ib,Error);ib.prototype.name="CustomError";var jb=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};/*

 Copyright Google LLC
 SPDX-License-Identifier: Apache-2.0
*/
var kb=globalThis.trustedTypes,lb;function mb(){var a=null;if(!kb)return a;try{var b=function(c){return c};
a=kb.createPolicy("goog#html",{createHTML:b,createScript:b,createScriptURL:b})}catch(c){}return a}
function nb(){lb===void 0&&(lb=mb());return lb}
;function ob(a){this.h=a}
ob.prototype.toString=function(){return this.h+""};
function pb(a){var b=nb();a=b?b.createScriptURL(a):a;return new ob(a)}
function qb(a){if(a instanceof ob)return a.h;throw Error("");}
;var rb=pa([""]),sb=qa(["\x00"],["\\0"]),tb=qa(["\n"],["\\n"]),ub=qa(["\x00"],["\\u0000"]);function vb(a){return a.toString().indexOf("`")===-1}
vb(function(a){return a(rb)})||vb(function(a){return a(sb)})||vb(function(a){return a(tb)})||vb(function(a){return a(ub)});function wb(a){this.h=a}
wb.prototype.toString=function(){return this.h};
var xb=new wb("about:invalid#zClosurez");function yb(a){this.If=a}
function zb(a){return new yb(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var Ab=[zb("data"),zb("http"),zb("https"),zb("mailto"),zb("ftp"),new yb(function(a){return/^[^:]*([/?#]|$)/.test(a)})],Bb=/^\s*(?!javascript:)(?:[\w+.-]+:|[^:/?#]*(?:[/?#]|$))/i;
function Cb(a){if(a instanceof wb)if(a instanceof wb)a=a.h;else throw Error("");else a=Bb.test(a)?a:void 0;return a}
;function Db(a,b){b=Cb(b);b!==void 0&&(a.href=b)}
;function Eb(a,b){throw Error(b===void 0?"unexpected value "+a+"!":b);}
;function Fb(a){this.h=a}
Fb.prototype.toString=function(){return this.h+""};function Gb(a){a=a===void 0?document:a;var b,c;a=(c=(b=a).querySelector)==null?void 0:c.call(b,"script[nonce]");return a==null?"":a.nonce||a.getAttribute("nonce")||""}
;function Hb(a){this.h=a}
Hb.prototype.toString=function(){return this.h+""};
function Ib(a){var b=nb();a=b?b.createScript(a):a;return new Hb(a)}
function Jb(a){if(a instanceof Hb)return a.h;throw Error("");}
;function Kb(a){var b=Gb(a.ownerDocument);b&&a.setAttribute("nonce",b)}
function Lb(a,b){a.src=qb(b);Kb(a)}
;function Mb(){this.h=Ob[0].toLowerCase()}
Mb.prototype.toString=function(){return this.h};function Pb(a){var b="true".toString(),c=[new Mb];if(c.length===0)throw Error("");if(c.map(function(d){if(d instanceof Mb)d=d.h;else throw Error("");return d}).every(function(d){return"data-loaded".indexOf(d)!==0}))throw Error('Attribute "data-loaded" does not match any of the allowed prefixes.');
a.setAttribute("data-loaded",b)}
;var Qb="alternate author bookmark canonical cite help icon license modulepreload next prefetch dns-prefetch prerender preconnect preload prev search subresource".split(" ");function Rb(a,b){if(b instanceof ob)a.href=qb(b).toString(),a.rel="stylesheet";else{if(Qb.indexOf("stylesheet")===-1)throw Error('TrustedResourceUrl href attribute required with rel="stylesheet"');b=Cb(b);b!==void 0&&(a.href=b,a.rel="stylesheet")}}
;var Sb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if(typeof a==="string")return typeof b!=="string"||b.length!=1?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},Tb=Array.prototype.forEach?function(a,b){Array.prototype.forEach.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=typeof a==="string"?a.split(""):a,e=0;e<c;e++)e in d&&b.call(void 0,d[e],e,a)},Ub=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f=typeof a==="string"?a.split(""):a,g=0;g<c;g++)if(g in f){var h=f[g];
b.call(void 0,h,g,a)&&(d[e++]=h)}return d},Vb=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e=typeof a==="string"?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},Wb=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
Tb(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function Xb(a,b){a:{for(var c=a.length,d=typeof a==="string"?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return b<0?null:typeof a==="string"?a.charAt(b):a[b]}
function Yb(a,b){b=Sb(a,b);var c;(c=b>=0)&&Array.prototype.splice.call(a,b,1);return c}
function Zb(a){var b=a.length;if(b>0){for(var c=Array(b),d=0;d<b;d++)c[d]=a[d];return c}return[]}
function $b(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Oa(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
function ac(a,b){return a>b?1:a<b?-1:0}
;function bc(a,b){a.__closure__error__context__984382||(a.__closure__error__context__984382={});a.__closure__error__context__984382.severity=b}
;function cc(a){var b=G("window.location.href");a==null&&(a='Unknown Error of type "null/undefined"');if(typeof a==="string")return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||D.$googDebugFname||b}catch(g){e="Not available",c=!0}b=dc(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(c==
null){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,ec[c])c=ec[c];else{c=String(c);if(!ec[c]){var f=/function\s+([^\(]+)/m.exec(c);ec[c]=f?f[1]:"[Anonymous]"}c=ec[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";typeof a.toString==="function"&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}return{message:a.message,
name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:b}}
function dc(a,b){b||(b={});b[fc(a)]=!0;var c=a.stack||"",d=a.cause;d&&!b[fc(d)]&&(c+="\nCaused by: ",d.stack&&d.stack.indexOf(d.toString())==0||(c+=typeof d==="string"?d:d.message+"\n"),c+=dc(d,b));a=a.errors;if(Array.isArray(a)){d=1;var e;for(e=0;e<a.length&&!(d>4);e++)b[fc(a[e])]||(c+="\nInner error "+d++ +": ",a[e].stack&&a[e].stack.indexOf(a[e].toString())==0||(c+=typeof a[e]==="string"?a[e]:a[e].message+"\n"),c+=dc(a[e],b));e<a.length&&(c+="\n... "+(a.length-e)+" more inner errors")}return c}
function fc(a){var b="";typeof a.toString==="function"&&(b=""+a);return b+a.stack}
var ec={};function hc(a){return decodeURIComponent(a.replace(/\+/g," "))}
function ic(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var jc=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function kc(a){return a?decodeURI(a):a}
function lc(a){return kc(a.match(jc)[3]||null)}
function mc(a){return kc(a.match(jc)[5]||null)}
function nc(a){var b=a.match(jc);a=b[5];var c=b[6];b=b[7];var d="";a&&(d+=a);c&&(d+="?"+c);b&&(d+="#"+b);return d}
function oc(a){var b=a.indexOf("#");return b<0?a:a.slice(0,b)}
function pc(a,b){if(a){a=a.split("&");for(var c=0;c<a.length;c++){var d=a[c].indexOf("="),e=null;if(d>=0){var f=a[c].substring(0,d);e=a[c].substring(d+1)}else f=a[c];b(f,e?hc(e):"")}}}
function qc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)qc(a,String(b[d]),c);else b!=null&&c.push(a+(b===""?"":"="+encodeURIComponent(String(b))))}
function rc(a){var b=[],c;for(c in a)qc(c,a[c],b);return b.join("&")}
function sc(a,b){b=rc(b);if(b){var c=a.indexOf("#");c<0&&(c=a.length);var d=a.indexOf("?");if(d<0||d>c){d=c;var e=""}else e=a.substring(d+1,c);a=[a.slice(0,d),e,a.slice(c)];c=a[1];a[1]=b?c?c+"&"+b:b:c;b=a[0]+(a[1]?"?"+a[1]:"")+a[2]}else b=a;return b}
function tc(a,b,c,d){for(var e=c.length;(b=a.indexOf(c,b))>=0&&b<d;){var f=a.charCodeAt(b-1);if(f==38||f==63)if(f=a.charCodeAt(b+e),!f||f==61||f==38||f==35)return b;b+=e+1}return-1}
var uc=/#|$/,vc=/[?&]($|#)/;function wc(a,b){for(var c=a.search(uc),d=0,e,f=[];(e=tc(a,d,b,c))>=0;)f.push(a.substring(d,e)),d=Math.min(a.indexOf("&",e)+1||c,c);f.push(a.slice(d));return f.join("").replace(vc,"$1")}
;function xc(){try{var a,b;return!!((a=window)==null?0:(b=a.top)==null?0:b.location.href)&&!1}catch(c){return!0}}
;function I(a,b,c){c=c===void 0?Error():c;var d=Error.call(this);this.message=d.message;"stack"in d&&(this.stack=d.stack);this.code=a;b+=":";c instanceof Error?(this.message=b+c.message,this.stack=c.stack||""):(this.message=b+String(c),this.stack="");Object.setPrototypeOf(this,this.constructor.prototype)}
v(I,Error);function yc(a){a&&typeof a.dispose=="function"&&a.dispose()}
;function zc(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];Oa(d)?zc.apply(null,d):yc(d)}}
;function J(){this.I=this.I;this.H=this.H}
J.prototype.I=!1;J.prototype.dispose=function(){this.I||(this.I=!0,this.X())};
J.prototype[Symbol.dispose]=function(){this.dispose()};
function Ac(a,b){a.addOnDisposeCallback(Ya(yc,b))}
J.prototype.addOnDisposeCallback=function(a,b){this.I?b!==void 0?a.call(b):a():(this.H||(this.H=[]),b&&(a=a.bind(b)),this.H.push(a))};
J.prototype.X=function(){if(this.H)for(;this.H.length;)this.H.shift()()};function Bc(){var a=Cc();a=a===void 0?"bevasrsg":a;return new Promise(function(b){var c=window===window.top?window:xc()?window:window.top,d=c[a],e;((e=d)==null?0:e.bevasrs)?b(new Dc(d.bevasrs)):(d||(d={},d=(d.nqfbel=[],d),c[a]=d),d.nqfbel.push(function(f){b(new Dc(f))}))})}
function Dc(a){J.call(this);var b=this;this.vm=a;this.i="keydown keypress keyup input focusin focusout select copy cut paste change click dblclick auxclick pointerover pointerdown pointerup pointermove pointerout dragenter dragleave drag dragend mouseover mousedown mouseup mousemove mouseout touchstart touchend touchmove wheel".split(" ");this.h=void 0;this.Pb=this.vm.p;this.j=this.lc.bind(this);this.addOnDisposeCallback(function(){return void Ec(b)})}
v(Dc,J);Dc.prototype.snapshot=function(a){return this.vm.s(Object.assign({},a.Ha&&{c:a.Ha},a.Zc&&{s:a.Zc},a.Ed!==void 0&&{p:a.Ed}))};
Dc.prototype.lc=function(a){this.vm.e(a)};
Dc.prototype.Dc=function(a,b){return this.vm.c(a,b,!1)};
function Ec(a){a.h!==void 0&&(a.i.forEach(function(b){var c;(c=a.h)==null||c.removeEventListener(b,a.j)}),a.h=void 0)}
Dc.prototype.uc=function(){return this.vm.l()};function Fc(a){var b,c,d={Ha:a.c,ld:a.e,Rf:(b=a.mc)!=null?b:!1,Sf:(c=a.me)!=null?c:!1};a.co&&(d.Ac={Od:a.co.c,Pe:a.co.a,zg:a.co.s});return d}
function Gc(a){return function(){var b;return B(function(c){if(c.h==1)return c.yield(a(),2);b=c.i;return c.return({f:function(){return b.Ob.promise},
c:function(d){if(d>150)var e=!1;else try{b.cache=new Hc(d,b.logger),e=!0}catch(f){Ic(b,new I(22,"GBJ:init",f)),e=!1}return e},
m:function(d){return b.gb(Fc(d))},
mws:function(d){return b.Oc(Fc(d))}})})}}
function Jc(a,b){var c=Cc();c=c===void 0?"bevasrsg":c;b={s:function(f){var g;return a.snapshot(Object.assign({},f.c&&{Ha:f.c},f.s&&{Zc:f.s},{Dj:(g=f.p)!=null?g:!0}))},
e:function(f){var g;return void((g=a.lc)==null?void 0:g.call(a,f))},
c:function(f,g){return a.Dc(f,g)},
p:a.Pb,l:function(){return a.uc()},
wpc:b?Gc(b):void 0};var d=window===window.top?window:xc()?window:window.top,e=d[c];if(e){e.bevasrs=b;if(e.nqfbel!==void 0)for(c=y(e.nqfbel),d=c.next();!d.done;d=c.next())d=d.value,d(b);e.nqfbel=void 0}else e={},e=(e.bevasrs=b,e.nqfbel=void 0,e),d[c]=e}
;function Kc(a){var b=b===void 0?52:b;var c=[];Lc(a,Mc,6).forEach(function(d){Nc(d,2)<=b&&c.push(Nc(d,1))});
return c}
function Oc(a){var b=b===void 0?52:b;var c=[];Lc(a,Mc,6).forEach(function(d){Nc(d,2)>b&&c.push(Nc(d,1))});
return c}
;function Pc(a){a.then(function(){},function(){})}
function Qc(){J.apply(this,arguments);this.i=1}
v(Qc,J);Qc.prototype.share=function(){if(this.I)throw Error("E:AD");this.i++;return this};
Qc.prototype.dispose=function(){--this.i||J.prototype.dispose.call(this)};function Rc(a){return{fieldType:2,fieldName:a}}
function Sc(a){return{fieldType:3,fieldName:a}}
;function Tc(a){this.h=a;a.hd("/client_streamz/bg/frs",Sc("mk"))}
Tc.prototype.record=function(a,b){this.h.record("/client_streamz/bg/frs",a,b)};
function Uc(a){this.h=a;a.hd("/client_streamz/bg/wrl",Sc("mn"),Rc("ac"),Rc("sc"),Sc("rk"),Sc("mk"))}
Uc.prototype.record=function(a,b,c,d,e,f){this.h.record("/client_streamz/bg/wrl",a,b,c,d,e,f)};
function Vc(a){this.h=a;a.Wb("/client_streamz/bg/ec",Sc("en"),Sc("mk"))}
Vc.prototype.Ia=function(a,b){this.h.Tb("/client_streamz/bg/ec",a,b)};
function Wc(a){this.h=a;a.hd("/client_streamz/bg/el",Sc("en"),Sc("mk"))}
Wc.prototype.record=function(a,b,c){this.h.record("/client_streamz/bg/el",a,b,c)};
function Xc(a){this.h=a;a.Wb("/client_streamz/bg/cec",Rc("ec"),Sc("mk"))}
Xc.prototype.Ia=function(a,b){this.h.Tb("/client_streamz/bg/cec",a,b)};
function Yc(a){this.h=a;a.Wb("/client_streamz/bg/po/csc",Rc("cs"),Sc("mk"))}
Yc.prototype.Ia=function(a,b){this.h.Tb("/client_streamz/bg/po/csc",a,b)};
function Zc(a){this.h=a;a.Wb("/client_streamz/bg/po/ctav",Sc("av"),Sc("mk"))}
Zc.prototype.Ia=function(a,b){this.h.Tb("/client_streamz/bg/po/ctav",a,b)};
function $c(a){this.h=a;a.Wb("/client_streamz/bg/po/cwsc",Sc("su"),Sc("mk"))}
$c.prototype.Ia=function(a,b){this.h.Tb("/client_streamz/bg/po/cwsc",a,b)};var ad,bd=typeof String.prototype.isWellFormed==="function",cd=typeof TextEncoder!=="undefined";
function dd(a){var b=!1;b=b===void 0?!1:b;if(cd){if(b&&(bd?!a.isWellFormed():/(?:[^\uD800-\uDBFF]|^)[\uDC00-\uDFFF]|[\uD800-\uDBFF](?![\uDC00-\uDFFF])/.test(a)))throw Error("Found an unpaired surrogate");a=(ad||(ad=new TextEncoder)).encode(a)}else{for(var c=0,d=new Uint8Array(3*a.length),e=0;e<a.length;e++){var f=a.charCodeAt(e);if(f<128)d[c++]=f;else{if(f<2048)d[c++]=f>>6|192;else{if(f>=55296&&f<=57343){if(f<=56319&&e<a.length){var g=a.charCodeAt(++e);if(g>=56320&&g<=57343){f=(f-55296)*1024+g-56320+
65536;d[c++]=f>>18|240;d[c++]=f>>12&63|128;d[c++]=f>>6&63|128;d[c++]=f&63|128;continue}else e--}if(b)throw Error("Found an unpaired surrogate");f=65533}d[c++]=f>>12|224;d[c++]=f>>6&63|128}d[c++]=f&63|128}}a=c===d.length?d:d.subarray(0,c)}return a}
;function ed(a){D.setTimeout(function(){throw a;},0)}
;function fd(a){for(var b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);e<128?b[c++]=e:(e<2048?b[c++]=e>>6|192:((e&64512)==55296&&d+1<a.length&&(a.charCodeAt(d+1)&64512)==56320?(e=65536+((e&1023)<<10)+(a.charCodeAt(++d)&1023),b[c++]=e>>18|240,b[c++]=e>>12&63|128):b[c++]=e>>12|224,b[c++]=e>>6&63|128),b[c++]=e&63|128)}return b}
;var gd=Ma(610401301,!1),hd=Ma(748402147,!0),id=Ma(824656860,!1);function jd(){var a=D.navigator;return a&&(a=a.userAgent)?a:""}
var kd,ld=D.navigator;kd=ld?ld.userAgentData||null:null;function md(a){if(!gd||!kd)return!1;for(var b=0;b<kd.brands.length;b++){var c=kd.brands[b].brand;if(c&&c.indexOf(a)!=-1)return!0}return!1}
function K(a){return jd().indexOf(a)!=-1}
;function nd(){return gd?!!kd&&kd.brands.length>0:!1}
function od(){return nd()?!1:K("Opera")}
function pd(){return K("Firefox")||K("FxiOS")}
function qd(){return nd()?md("Chromium"):(K("Chrome")||K("CriOS"))&&!(nd()?0:K("Edge"))||K("Silk")}
;function rd(){return gd?!!kd&&!!kd.platform:!1}
function sd(){return K("iPhone")&&!K("iPod")&&!K("iPad")}
;function td(a){td[" "](a);return a}
td[" "]=function(){};var ud=od(),vd=nd()?!1:K("Trident")||K("MSIE"),wd=K("Edge"),xd=K("Gecko")&&!(jd().toLowerCase().indexOf("webkit")!=-1&&!K("Edge"))&&!(K("Trident")||K("MSIE"))&&!K("Edge"),yd=jd().toLowerCase().indexOf("webkit")!=-1&&!K("Edge");yd&&K("Mobile");rd()||K("Macintosh");rd()||K("Windows");(rd()?kd.platform==="Linux":K("Linux"))||rd()||K("CrOS");var zd=rd()?kd.platform==="Android":K("Android");sd();K("iPad");K("iPod");sd()||K("iPad")||K("iPod");jd().toLowerCase().indexOf("kaios");pd();var Ad=sd()||K("iPod"),Bd=K("iPad");!K("Android")||qd()||pd()||od()||K("Silk");qd();var Cd=K("Safari")&&!(qd()||(nd()?0:K("Coast"))||od()||(nd()?0:K("Edge"))||(nd()?md("Microsoft Edge"):K("Edg/"))||(nd()?md("Opera"):K("OPR"))||pd()||K("Silk")||K("Android"))&&!(sd()||K("iPad")||K("iPod"));var Dd={},Ed=null;function Fd(a,b){Oa(a);b===void 0&&(b=0);Gd();b=Dd[b];for(var c=Array(Math.floor(a.length/3)),d=b[64]||"",e=0,f=0;e<a.length-2;e+=3){var g=a[e],h=a[e+1],k=a[e+2],l=b[g>>2];g=b[(g&3)<<4|h>>4];h=b[(h&15)<<2|k>>6];k=b[k&63];c[f++]=""+l+g+h+k}l=0;k=d;switch(a.length-e){case 2:l=a[e+1],k=b[(l&15)<<2]||d;case 1:a=a[e],c[f]=""+b[a>>2]+b[(a&3)<<4|l>>4]+k+d}return c.join("")}
function Hd(a){var b=a.length,c=b*3/4;c%3?c=Math.floor(c):"=.".indexOf(a[b-1])!=-1&&(c="=.".indexOf(a[b-2])!=-1?c-2:c-1);var d=new Uint8Array(c),e=0;Id(a,function(f){d[e++]=f});
return e!==c?d.subarray(0,e):d}
function Id(a,b){function c(k){for(;d<a.length;){var l=a.charAt(d++),m=Ed[l];if(m!=null)return m;if(!/^[\s\xa0]*$/.test(l))throw Error("Unknown base64 encoding at char: "+l);}return k}
Gd();for(var d=0;;){var e=c(-1),f=c(0),g=c(64),h=c(64);if(h===64&&e===-1)break;b(e<<2|f>>4);g!=64&&(b(f<<4&240|g>>2),h!=64&&b(g<<6&192|h))}}
function Gd(){if(!Ed){Ed={};for(var a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),b=["+/=","+/","-_=","-_.","-_"],c=0;c<5;c++){var d=a.concat(b[c].split(""));Dd[c]=d;for(var e=0;e<d.length;e++){var f=d[e];Ed[f]===void 0&&(Ed[f]=e)}}}}
;var Jd=typeof Uint8Array!=="undefined",Kd=!vd&&typeof btoa==="function",Ld=/[-_.]/g,Md={"-":"+",_:"/",".":"="};function Nd(a){return Md[a]||""}
var Pd={};function Qd(a,b){Rd(b);this.h=a;if(a!=null&&a.length===0)throw Error("ByteString should be constructed with non-empty values");}
function Sd(){return Td||(Td=new Qd(null,Pd))}
function Ud(a){return new Uint8Array(Vd(a)||0)}
Qd.prototype.sizeBytes=function(){var a=Vd(this);return a?a.length:0};
function Vd(a){Rd(Pd);var b=a.h;if(!(b==null||Jd&&b!=null&&b instanceof Uint8Array))if(typeof b==="string")if(Kd){b=Ld.test(b)?b.replace(Ld,Nd):b;b=atob(b);for(var c=new Uint8Array(b.length),d=0;d<b.length;d++)c[d]=b.charCodeAt(d);b=c}else b=Hd(b);else Na(b),b=null;return b==null?b:a.h=b}
var Td;function Rd(a){if(a!==Pd)throw Error("illegal external caller");}
;var Wd=void 0;function Xd(a){a=Error(a);bc(a,"warning");return a}
function Yd(a,b){if(a!=null){var c;var d=(c=Wd)!=null?c:Wd={};c=d[a]||0;c>=b||(d[a]=c+1,a=Error(),bc(a,"incident"),ed(a))}}
;var Zd=typeof Symbol==="function"&&typeof Symbol()==="symbol";function $d(a,b,c){return typeof Symbol==="function"&&typeof Symbol()==="symbol"?(c===void 0?0:c)&&Symbol.for&&a?Symbol.for(a):a!=null?Symbol(a):Symbol():b}
var ae=$d("jas",void 0,!0),be=$d(void 0,"1oa"),ce=$d(void 0,Symbol()),de=$d(void 0,"0ub"),ee=$d(void 0,"0ubs"),fe=$d(void 0,"0ubsb"),ge=$d(void 0,"0actk"),he=$d("m_m","nj",!0),ie=$d(void 0,"vps"),je=$d();Math.max.apply(Math,A(Object.values({Ki:1,Ji:2,Ii:4,Oi:8,Qi:16,Mi:32,Vg:64,Gi:128,ah:256,Pi:512,bh:1024,Hi:2048,Ni:4096,Li:8192})));var ke={Gf:{value:0,configurable:!0,writable:!0,enumerable:!1}},le=Object.defineProperties,L=Zd?ae:"Gf",me,ne=[];oe(ne,7);me=Object.freeze(ne);function pe(a,b){Zd||L in a||le(a,ke);a[L]|=b}
function oe(a,b){Zd||L in a||le(a,ke);a[L]=b}
;function qe(){return typeof BigInt==="function"}
;var re={};function se(a,b){return b===void 0?a.h!==te&&!!(2&(a.D[L]|0)):!!(2&b)&&a.h!==te}
var te={};function ue(a,b){if(a!=null)if(typeof a==="string")a=a?new Qd(a,Pd):Sd();else if(a.constructor!==Qd)if(Jd&&a!=null&&a instanceof Uint8Array)a instanceof Uint8Array||Array.isArray(a),a=a.length?new Qd(new Uint8Array(a),Pd):Sd();else{if(!b)throw Error();a=void 0}return a}
var ve=Object.freeze({});function we(a,b,c){var d=b&128?0:-1,e=a.length,f;if(f=!!e)f=a[e-1],f=f!=null&&typeof f==="object"&&f.constructor===Object;var g=e+(f?-1:0);for(b=b&128?1:0;b<g;b++)c(b-d,a[b]);if(f){a=a[e-1];for(var h in a)!isNaN(h)&&c(+h,a[h])}}
var xe={};function ye(a){a.fj=!0;return a}
;var ze=ye(function(a){return typeof a==="number"}),Ae=ye(function(a){return typeof a==="string"}),Be=ye(function(a){return typeof a==="boolean"});
function Ce(){var a=De;return ye(function(b){for(var c in a)if(b===a[c]&&!/^[0-9]+$/.test(c))return!0;return!1})}
var Ee=ye(function(a){return a!=null&&typeof a==="object"&&typeof a.then==="function"}),Fe=ye(function(a){return!!a&&(typeof a==="object"||typeof a==="function")});var Ge=typeof D.BigInt==="function"&&typeof D.BigInt(0)==="bigint";function He(a){var b=a;if(Ae(b)){if(!/^\s*(?:-?[1-9]\d*|0)?\s*$/.test(b))throw Error(String(b));}else if(ze(b)&&!Number.isSafeInteger(b))throw Error(String(b));return Ge?BigInt(a):a=Be(a)?a?"1":"0":Ae(a)?a.trim()||"0":String(a)}
var Ne=ye(function(a){return Ge?a>=Ie&&a<=Je:a[0]==="-"?Ke(a,Le):Ke(a,Me)}),Le=Number.MIN_SAFE_INTEGER.toString(),Ie=Ge?BigInt(Number.MIN_SAFE_INTEGER):void 0,Me=Number.MAX_SAFE_INTEGER.toString(),Je=Ge?BigInt(Number.MAX_SAFE_INTEGER):void 0;
function Ke(a,b){if(a.length>b.length)return!1;if(a.length<b.length||a===b)return!0;for(var c=0;c<a.length;c++){var d=a[c],e=b[c];if(d>e)return!1;if(d<e)return!0}}
;var Oe=0,Pe=0,Qe;function Re(a){var b=a>>>0;Oe=b;Pe=(a-b)/4294967296>>>0}
function Se(a){if(a<0){Re(0-a);var b=y(Te(Oe,Pe));a=b.next().value;b=b.next().value;Oe=a>>>0;Pe=b>>>0}else Re(a)}
function Ue(a,b){var c=b*4294967296+(a>>>0);return Number.isSafeInteger(c)?c:Ve(a,b)}
function Ve(a,b){b>>>=0;a>>>=0;if(b<=2097151)var c=""+(4294967296*b+a);else qe()?c=""+(BigInt(b)<<BigInt(32)|BigInt(a)):(c=(a>>>24|b<<8)&16777215,b=b>>16&65535,a=(a&16777215)+c*6777216+b*6710656,c+=b*8147497,b*=2,a>=1E7&&(c+=a/1E7>>>0,a%=1E7),c>=1E7&&(b+=c/1E7>>>0,c%=1E7),c=b+We(c)+We(a));return c}
function We(a){a=String(a);return"0000000".slice(a.length)+a}
function Xe(){var a=Oe,b=Pe;b&2147483648?qe()?a=""+(BigInt(b|0)<<BigInt(32)|BigInt(a>>>0)):(b=y(Te(a,b)),a=b.next().value,b=b.next().value,a="-"+Ve(a,b)):a=Ve(a,b);return a}
function Ye(a){if(a.length<16)Se(Number(a));else if(qe())a=BigInt(a),Oe=Number(a&BigInt(4294967295))>>>0,Pe=Number(a>>BigInt(32)&BigInt(4294967295));else{var b=+(a[0]==="-");Pe=Oe=0;for(var c=a.length,d=0+b,e=(c-b)%6+b;e<=c;d=e,e+=6)d=Number(a.slice(d,e)),Pe*=1E6,Oe=Oe*1E6+d,Oe>=4294967296&&(Pe+=Math.trunc(Oe/4294967296),Pe>>>=0,Oe>>>=0);b&&(b=y(Te(Oe,Pe)),a=b.next().value,b=b.next().value,Oe=a,Pe=b)}}
function Te(a,b){b=~b;a?a=~a+1:b+=1;return[a,b]}
;function Ze(a){return Array.prototype.slice.call(a)}
;var $e=typeof BigInt==="function"?BigInt.asIntN:void 0,af=typeof BigInt==="function"?BigInt.asUintN:void 0,bf=Number.isSafeInteger,cf=Number.isFinite,df=Math.trunc;function ef(a){return a.displayName||a.name||"unknown type name"}
function ff(a){if(a!=null&&typeof a!=="boolean")throw Error("Expected boolean but got "+Na(a)+": "+a);return a}
var gf=/^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;function hf(a){switch(typeof a){case "bigint":return!0;case "number":return cf(a);case "string":return gf.test(a);default:return!1}}
function jf(a){if(typeof a!=="number")throw Xd("int32");if(!cf(a))throw Xd("int32");return a|0}
function kf(a){return a==null?a:jf(a)}
function lf(a){if(a==null)return a;if(typeof a==="string"&&a)a=+a;else if(typeof a!=="number")return;return cf(a)?a|0:void 0}
function mf(a){if(a==null)return a;if(typeof a==="string"&&a)a=+a;else if(typeof a!=="number")return;return cf(a)?a>>>0:void 0}
function nf(a){var b=0;b=b===void 0?0:b;if(!hf(a))throw Xd("int64");var c=typeof a;switch(b){case 512:switch(c){case "string":return of(a);case "bigint":return String($e(64,a));default:return pf(a)}case 1024:switch(c){case "string":return qf(a);case "bigint":return He($e(64,a));default:return rf(a)}case 0:switch(c){case "string":return of(a);case "bigint":return He($e(64,a));default:return sf(a)}default:return Eb(b,"Unknown format requested type for int64")}}
function tf(a){return a==null?a:nf(a)}
function uf(a){a.indexOf(".");var b=a.length;if(a[0]==="-"?b<20||b===20&&a<="-9223372036854775808":b<19||b===19&&a<="9223372036854775807")return a;Ye(a);return Xe()}
function sf(a){hf(a);a=df(a);if(!bf(a)){Se(a);var b=Oe,c=Pe;if(a=c&2147483648)b=~b+1>>>0,c=~c>>>0,b==0&&(c=c+1>>>0);b=Ue(b,c);a=typeof b==="number"?a?-b:b:a?"-"+b:b}return a}
function pf(a){hf(a);a=df(a);bf(a)?a=String(a):(Se(a),a=Xe());return a}
function of(a){hf(a);var b=df(Number(a));if(bf(b))return String(b);b=a.indexOf(".");b!==-1&&(a=a.substring(0,b));return uf(a)}
function qf(a){var b=df(Number(a));if(bf(b))return He(b);b=a.indexOf(".");b!==-1&&(a=a.substring(0,b));return qe()?He($e(64,BigInt(a))):He(uf(a))}
function rf(a){return bf(a)?He(sf(a)):He(pf(a))}
function vf(a){if(a==null)return a;if(typeof a==="bigint")return Ne(a)?a=Number(a):(a=$e(64,a),a=Ne(a)?Number(a):String(a)),a;if(hf(a))return typeof a==="number"?sf(a):of(a)}
function wf(a){var b=typeof a;if(a==null)return a;if(b==="bigint")return He($e(64,a));if(hf(a))return b==="string"?qf(a):rf(a)}
function xf(a){if(a==null)return a;var b=typeof a;if(b==="bigint")return String($e(64,a));if(hf(a)){if(b==="string")return of(a);if(b==="number")return sf(a)}}
function yf(a){if(a==null)return a;var b=typeof a;if(b==="bigint")return String(af(64,a));if(hf(a)){if(b==="string")return hf(a),b=df(Number(a)),bf(b)&&b>=0?a=String(b):(b=a.indexOf("."),b!==-1&&(a=a.substring(0,b)),a.indexOf("."),a[0]==="-"?b=!1:(b=a.length,b=b<20?!0:b===20&&a<="18446744073709551615"),b||(Ye(a),a=Ve(Oe,Pe))),a;if(b==="number")return hf(a),a=df(a),a>=0&&bf(a)||(Se(a),a=Ue(Oe,Pe)),a}}
function zf(a){if(typeof a!=="string")throw Error();return a}
function Af(a){if(a!=null&&typeof a!=="string")throw Error();return a}
function Bf(a){return a==null||typeof a==="string"?a:void 0}
function Cf(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+ef(b)+" but got "+(a&&ef(a.constructor)));}
function Df(a,b,c){if(a!=null&&a[he]===re)return a;if(Array.isArray(a)){var d=a[L]|0;c=d|c&32|c&2;c!==d&&oe(a,c);return new b(a)}}
;var Ef={};function Ff(a){return a}
;function Gf(a){var b=$a(ce);return b?a[b]:void 0}
var Hf={vj:!0};function If(a,b){b<100||Yd(ee,1)}
;function Jf(a,b,c,d){var e=d!==void 0;d=!!d;var f=$a(ce),g;!e&&Zd&&f&&(g=a[f])&&g.tf(If);f=[];var h=a.length;g=4294967295;var k=!1,l=!!(b&64),m=l?b&128?0:-1:void 0;if(!(b&1)){var n=h&&a[h-1];n!=null&&typeof n==="object"&&n.constructor===Object?(h--,g=h):n=void 0;if(l&&!(b&128)&&!e){k=!0;var r;g=((r=Kf)!=null?r:Ff)(g-m,m,a,n,void 0)+m}}b=void 0;for(r=0;r<h;r++){var t=a[r];if(t!=null&&(t=c(t,d))!=null)if(l&&r>=g){var w=r-m,z=void 0;((z=b)!=null?z:b={})[w]=t}else f[r]=t}if(n)for(var x in n)h=n[x],h!=
null&&(h=c(h,d))!=null&&(r=+x,t=void 0,l&&!Number.isNaN(r)&&(t=r+m)<g?f[t]=h:(r=void 0,((r=b)!=null?r:b={})[x]=h));b&&(k?f.push(b):f[g]=b);e&&$a(ce)&&Gf(a);return f}
function Lf(a){switch(typeof a){case "number":return Number.isFinite(a)?a:""+a;case "bigint":return Ne(a)?Number(a):""+a;case "boolean":return a?1:0;case "object":if(Array.isArray(a)){var b=a[L]|0;return a.length===0&&b&1?void 0:Jf(a,b,Lf)}if(a!=null&&a[he]===re)return Mf(a);if(a instanceof Qd){b=a.h;if(b==null)a="";else if(typeof b==="string")a=b;else{if(Kd){for(var c="",d=0,e=b.length-10240;d<e;)c+=String.fromCharCode.apply(null,b.subarray(d,d+=10240));c+=String.fromCharCode.apply(null,d?b.subarray(d):
b);b=btoa(c)}else b=Fd(b);a=a.h=b}return a}return}return a}
var Kf;function Nf(a,b){if(b){Kf=b==null||b===Ff||b[ie]!==Ef?Ff:b;try{return Mf(a)}finally{Kf=void 0}}return Mf(a)}
function Mf(a){a=a.D;return Jf(a,a[L]|0,Lf)}
;var Of,Pf;function Qf(a){switch(typeof a){case "boolean":return Of||(Of=[0,void 0,!0]);case "number":return a>0?void 0:a===0?Pf||(Pf=[0,void 0]):[-a,void 0];case "string":return[0,a];case "object":return a}}
function M(a,b,c){return Rf(a,b,c,2048)}
function Rf(a,b,c,d){d=d===void 0?0:d;if(a==null){var e=32;c?(a=[c],e|=128):a=[];b&&(e=e&-16760833|(b&1023)<<14)}else{if(!Array.isArray(a))throw Error("narr");e=a[L]|0;if(hd&&1&e)throw Error("rfarr");2048&e&&!(2&e)&&Sf();if(e&256)throw Error("farr");if(e&64)return(e|d)!==e&&oe(a,e|d),a;if(c&&(e|=128,c!==a[0]))throw Error("mid");a:{c=a;e|=64;var f=c.length;if(f){var g=f-1,h=c[g];if(h!=null&&typeof h==="object"&&h.constructor===Object){b=e&128?0:-1;g-=b;if(g>=1024)throw Error("pvtlmt");for(var k in h)f=
+k,f<g&&(c[f+b]=h[k],delete h[k]);e=e&-16760833|(g&1023)<<14;break a}}if(b){k=Math.max(b,f-(e&128?0:-1));if(k>1024)throw Error("spvt");e=e&-16760833|(k&1023)<<14}}}oe(a,e|64|d);return a}
function Sf(){if(hd)throw Error("carr");Yd(ge,5)}
;function Tf(a,b){if(typeof a!=="object")return a;if(Array.isArray(a)){var c=a[L]|0;a.length===0&&c&1?a=void 0:c&2||(!b||4096&c||16&c?a=Uf(a,c,!1,b&&!(c&16)):(pe(a,34),c&4&&Object.freeze(a)));return a}if(a!=null&&a[he]===re)return b=a.D,c=b[L]|0,se(a,c)?a:Vf(a,b,c)?Wf(a,b):Uf(b,c);if(a instanceof Qd)return a}
function Wf(a,b,c){a=new a.constructor(b);c&&(a.h=te);a.i=te;return a}
function Uf(a,b,c,d){d!=null||(d=!!(34&b));a=Jf(a,b,Tf,d);d=32;c&&(d|=2);b=b&16769217|d;oe(a,b);return a}
function Xf(a){var b=a.D,c=b[L]|0;return se(a,c)?Vf(a,b,c)?Wf(a,b,!0):new a.constructor(Uf(b,c,!1)):a}
function Yf(a){if(a.h!==te)return!1;var b=a.D;b=Uf(b,b[L]|0);pe(b,2048);a.D=b;a.h=void 0;a.i=void 0;return!0}
function Zf(a){if(!Yf(a)&&se(a,a.D[L]|0))throw Error();}
function $f(a,b){b===void 0&&(b=a[L]|0);b&32&&!(b&4096)&&oe(a,b|4096)}
function Vf(a,b,c){return c&2?!0:c&32&&!(c&4096)?(oe(b,c|2),a.h=te,!0):!1}
;var ag=He(0),bg={};function cg(a,b,c,d,e){Object.isExtensible(a);b=dg(a.D,b,c,e);if(b!==null||d&&a.i!==te)return b}
function dg(a,b,c,d){if(b===-1)return null;var e=b+(c?0:-1),f=a.length-1;if(!(f<1+(c?0:-1))){if(e>=f){var g=a[f];if(g!=null&&typeof g==="object"&&g.constructor===Object){c=g[b];var h=!0}else if(e===f)c=g;else return}else c=a[e];if(d&&c!=null){d=d(c);if(d==null)return d;if(!Object.is(d,c))return h?g[b]=d:a[e]=d,d}return c}}
function eg(a,b,c,d){Zf(a);var e=a.D;fg(e,e[L]|0,b,c,d);return a}
function fg(a,b,c,d,e){var f=c+(e?0:-1),g=a.length-1;if(g>=1+(e?0:-1)&&f>=g){var h=a[g];if(h!=null&&typeof h==="object"&&h.constructor===Object)return h[c]=d,b}if(f<=g)return a[f]=d,b;if(d!==void 0){var k;g=((k=b)!=null?k:b=a[L]|0)>>14&1023||536870912;c>=g?d!=null&&(f={},a[g+(e?0:-1)]=(f[c]=d,f)):a[f]=d}return b}
function gg(a){return!!(2&a)&&!!(4&a)||!!(256&a)}
function hg(a){return ue(a,!0)}
function ig(a){a=cg(a,1,void 0,void 0,hg);return a==null?Sd():a}
function jg(a,b,c){Zf(a);var d=a.D,e=d[L]|0;if(b==null)return fg(d,e,3),a;if(!Array.isArray(b))throw Xd();var f=b===me?7:b[L]|0,g=f,h=gg(f),k=h||Object.isFrozen(b);h||(f=0);k||(b=Ze(b),g=0,f=kg(f,e),k=!1);f|=5;h=4&f?512&f?512:1024&f?1024:0:void 0;h=h!=null?h:0;for(var l=0;l<b.length;l++){var m=b[l],n=c(m,h);Object.is(m,n)||(k&&(b=Ze(b),g=0,f=kg(f,e),k=!1),b[l]=n)}f!==g&&(k&&(b=Ze(b),f=kg(f,e)),oe(b,f));fg(d,e,3,b);return a}
function lg(a,b,c,d){Zf(a);var e=a.D;fg(e,e[L]|0,b,c===""?void 0:c,d);return a}
function mg(a,b,c,d){Zf(a);a=a.D;var e=a[L]|0;if(d==null){var f=ng(a);if(og(f,a,e,c)===b)f.set(c,0);else return}else{b===0||c.includes(b);f=ng(a);var g=og(f,a,e,c);g!==b&&(g&&(e=fg(a,e,g)),f.set(c,b))}fg(a,e,b,d)}
function ng(a){if(Zd){var b;return(b=a[be])!=null?b:a[be]=new Map}if(be in a)return a[be];b=new Map;Object.defineProperty(a,be,{value:b});return b}
function og(a,b,c,d){var e=a.get(d);if(e!=null)return e;for(var f=e=0;f<d.length;f++){var g=d[f];dg(b,g)!=null&&(e!==0&&(c=fg(b,c,e)),e=g)}a.set(d,e);return e}
function pg(a,b,c,d,e){var f=!1;d=dg(a,d,e,function(g){var h=Df(g,c,b);f=h!==g&&h!=null;return h});
if(d!=null)return f&&!se(d)&&$f(a,b),d}
function qg(a,b,c,d){var e=a.D,f=e[L]|0;b=pg(e,f,b,c,d);if(b==null)return b;f=e[L]|0;if(!se(a,f)){var g=Xf(b);g!==b&&(Yf(a)&&(e=a.D,f=e[L]|0),b=g,f=fg(e,f,c,b,d),$f(e,f))}return b}
function Lc(a,b,c){var d=void 0===ve?2:4;var e=a.D,f=e,g=e[L]|0,h=se(a,g);e=h?1:d;d=e===3;var k=!h;(e===2||k)&&Yf(a)&&(f=a.D,g=f[L]|0);a=dg(f,c);h=Array.isArray(a)?a:me;var l=h===me?7:h[L]|0;a=l;2&g&&(a|=2);var m=a|1;if(a=!(4&m)){var n=h,r=g,t=!!(2&m);t&&(r|=2);for(var w=!t,z=!0,x=0,H=0;x<n.length;x++){var F=Df(n[x],b,r);if(F instanceof b){if(!t){var V=se(F);w&&(w=!V);z&&(z=V)}n[H++]=F}}H<x&&(n.length=H);m|=4;m=z?m&-4097:m|4096;m=w?m|8:m&-9}m!==l&&(oe(h,m),2&m&&Object.freeze(h));if(k&&!(8&m||!h.length&&
(e===1||(e!==4?0:2&m||!(16&m)&&32&g)))){gg(m)&&(h=Ze(h),m=kg(m,g),g=fg(f,g,c,h));b=h;k=m;for(l=0;l<b.length;l++)n=b[l],m=Xf(n),n!==m&&(b[l]=m);k|=8;m=k=b.length?k|4096:k&-4097;oe(h,m)}b=h;k=h=m;e===1||(e!==4?0:2&h||!(16&h)&&32&g)?gg(h)||(h|=!b.length||a&&!(4096&h)||32&g&&!(4096&h||16&h)?2:256,h!==k&&oe(b,h),Object.freeze(b)):(e===2&&gg(h)&&(b=Ze(b),k=0,h=kg(h,g),g=fg(f,g,c,b)),gg(h)||(d||(h|=16),h!==k&&oe(b,h)));2&h||!(4096&h||16&h)||$f(f,g);return b}
function rg(a,b){a!=null?Cf(a,b):a=void 0;return a}
function sg(a,b,c,d,e){d=rg(d,b);eg(a,c,d,e);d&&!se(d)&&$f(a.D);return a}
function tg(a,b,c,d){Zf(a);var e=a.D,f=e[L]|0;if(d==null)return fg(e,f,c),a;if(!Array.isArray(d))throw Xd();for(var g=d===me?7:d[L]|0,h=g,k=gg(g),l=k||Object.isFrozen(d),m=!0,n=!0,r=0;r<d.length;r++){var t=d[r];Cf(t,b);k||(t=se(t),m&&(m=!t),n&&(n=t))}k||(g=m?13:5,g=n?g&-4097:g|4096);l&&g===h||(d=Ze(d),h=0,g=kg(g,f));g!==h&&oe(d,g);f=fg(e,f,c,d);2&g||!(4096&g||16&g)||$f(e,f);return a}
function kg(a,b){return a=(2&b?a|2:a&-3)&-273}
function Nc(a,b,c){c=c===void 0?0:c;a=lf(cg(a,b));return a!=null?a:c}
function ug(a,b){var c=c===void 0?0:c;a=mf(cg(a,b));return a!=null?a:c}
function vg(a,b,c){c=c===void 0?ag:c;a=id?cg(a,b,void 0,void 0,wf):wf(cg(a,b));return a!=null?a:c}
function wg(a,b,c,d){c=c===void 0?"":c;var e;return(e=Bf(cg(a,b,d)))!=null?e:c}
function xg(a){var b=b===void 0?0:b;a=cg(a,1);a=a==null?a:cf(a)?a|0:void 0;return a!=null?a:b}
function yg(a,b,c){return eg(a,b,Af(c))}
function zg(a,b,c){if(c!=null){if(!cf(c))throw Xd("enum");c|=0}return eg(a,b,c)}
;function Ag(a,b){this.i=a>>>0;this.h=b>>>0}
function Bg(a){if(!a)return Cg||(Cg=new Ag(0,0));if(!/^\d+$/.test(a))return null;Ye(a);return new Ag(Oe,Pe)}
var Cg;function Dg(a,b){this.i=a>>>0;this.h=b>>>0}
function Eg(a){if(!a)return Fg||(Fg=new Dg(0,0));if(!/^-?\d+$/.test(a))return null;Ye(a);return new Dg(Oe,Pe)}
var Fg;function Gg(){this.h=[]}
Gg.prototype.length=function(){return this.h.length};
Gg.prototype.end=function(){var a=this.h;this.h=[];return a};
function Hg(a,b,c){for(;c>0||b>127;)a.h.push(b&127|128),b=(b>>>7|c<<25)>>>0,c>>>=7;a.h.push(b)}
function Ig(a,b){for(;b>127;)a.h.push(b&127|128),b>>>=7;a.h.push(b)}
Gg.prototype.writeUint8=function(a){this.h.push(a>>>0&255)};
function Jg(a,b){a.h.push(b>>>0&255);a.h.push(b>>>8&255);a.h.push(b>>>16&255);a.h.push(b>>>24&255)}
Gg.prototype.writeInt8=function(a){this.h.push(a>>>0&255)};function Kg(){this.j=[];this.i=0;this.h=new Gg}
function Lg(a,b){b.length!==0&&(a.j.push(b),a.i+=b.length)}
function Mg(a,b){Ig(a.h,b*8+2);b=a.h.end();Lg(a,b);b.push(a.i);return b}
function Ng(a,b){var c=b.pop();for(c=a.i+a.h.length()-c;c>127;)b.push(c&127|128),c>>>=7,a.i++;b.push(c);a.i++}
function Og(a,b,c){if(c!=null){switch(typeof c){case "string":Bg(c)}Ig(a.h,b*8+1);switch(typeof c){case "number":a=a.h;Re(c);Jg(a,Oe);Jg(a,Pe);break;case "bigint":c=BigInt.asUintN(64,c);c=new Ag(Number(c&BigInt(4294967295)),Number(c>>BigInt(32)));a=a.h;b=c.h;Jg(a,c.i);Jg(a,b);break;default:c=Bg(c),a=a.h,b=c.h,Jg(a,c.i),Jg(a,b)}}}
function Pg(a,b,c){Ig(a.h,b*8+2);Ig(a.h,c.length);Lg(a,a.h.end());Lg(a,c)}
;function Qg(){function a(){throw Error();}
Object.setPrototypeOf(a,a.prototype);return a}
var Rg=Qg(),Sg=Qg(),Tg=Qg(),Ug=Qg(),Vg=Qg(),Wg=Qg(),Xg=Qg();function N(a,b,c){this.D=M(a,b,c)}
N.prototype.toJSON=function(){return Nf(this)};
N.prototype.serialize=function(a){return JSON.stringify(Nf(this,a))};
function Yg(a,b){if(b==null||b=="")return new a;b=JSON.parse(b);if(!Array.isArray(b))throw Error("dnarr");pe(b,32);return new a(b)}
N.prototype.clone=function(){var a=this.D,b=a[L]|0;return Vf(this,a,b)?Wf(this,a,!0):new this.constructor(Uf(a,b,!1))};
N.prototype[he]=re;N.prototype.toString=function(){return this.D.toString()};function Zg(a,b){this.ed=a;a=$a(Rg);this.h=!!a&&b===a||!1}
function $g(a){var b=b===void 0?Rg:b;return new Zg(a,b)}
function ah(a,b,c,d,e){b=bh(b,d);b!=null&&(c=Mg(a,c),e(b,a),Ng(a,c))}
var ch=$g(ah),dh=$g(ah),eh=Symbol(),fh=Symbol(),gh,hh;
function ih(a){var b=jh,c=kh,d=a[eh];if(d)return d;d={};d.Ti=a;d.be=Qf(a[0]);var e=a[1],f=1;e&&e.constructor===Object&&(d.extensions=e,e=a[++f],typeof e==="function"&&(d.Hf=!0,gh!=null||(gh=e),hh!=null||(hh=a[f+1]),e=a[f+=2]));for(var g={};e&&lh(e);){for(var h=0;h<e.length;h++)g[e[h]]=e;e=a[++f]}for(h=1;e!==void 0;){typeof e==="number"&&(h+=e,e=a[++f]);var k=void 0;if(e instanceof Zg)var l=e;else l=ch,f--;e=void 0;if((e=l)==null?0:e.h){e=a[++f];k=a;var m=f;typeof e==="function"&&(e=e(),k[m]=e);k=
e}e=a[++f];m=h+1;typeof e==="number"&&e<0&&(m-=e,e=a[++f]);for(;h<m;h++){var n=g[h];k?c(d,h,l,k,n):b(d,h,l,n)}}return a[eh]=d}
function lh(a){return Array.isArray(a)&&!!a.length&&typeof a[0]==="number"&&a[0]>0}
function bh(a,b){if(a instanceof N)return a.D;if(Array.isArray(a))return Rf(a,b[0],b[1])}
;function jh(a,b,c){a[b]=c.ed}
function kh(a,b,c,d){var e,f,g=c.ed;a[b]=function(h,k,l){return g(h,k,l,f||(f=ih(d).be),e||(e=mh(d)))}}
function mh(a){var b=a[fh];if(!b){var c=ih(a);b=function(d,e){return nh(d,e,c)};
a[fh]=b}return b}
function nh(a,b,c){we(a,a[L]|0,function(d,e){if(e!=null){var f=oh(c,d);f?f(b,e,d):d<500||Yd(fe,3)}});
(a=Gf(a))&&a.tf(function(d,e,f){Lg(b,b.h.end());for(d=0;d<f.length;d++)Lg(b,Vd(f[d])||new Uint8Array(0))})}
function oh(a,b){var c=a[b];if(c)return c;if(c=a.extensions)if(c=c[b]){c=Array.isArray(c)?c[0]instanceof Zg?c:[dh,c]:[c,void 0];var d=c[0].ed;if(c=c[1]){var e=mh(c),f=ih(c).be;c=a.Hf?hh(f,e):function(g,h,k){return d(g,h,k,f,e)}}else c=d;
return a[b]=c}}
;function ph(a,b,c){if(Array.isArray(b)){var d=b[L]|0;if(d&4)return b;for(var e=0,f=0;e<b.length;e++){var g=a(b[e]);g!=null&&(b[f++]=g)}f<e&&(b.length=f);a=d|1;c&&(a=(a|4)&-1537);a!==d&&oe(b,a);c&&a&2&&Object.freeze(b);return b}}
function qh(a,b,c){b=b==null||typeof b==="number"?b:b==="NaN"||b==="Infinity"||b==="-Infinity"?Number(b):void 0;b!=null&&(Ig(a.h,c*8+1),a=a.h,c=Qe||(Qe=new DataView(new ArrayBuffer(8))),c.setFloat64(0,+b,!0),Oe=c.getUint32(0,!0),Pe=c.getUint32(4,!0),Jg(a,Oe),Jg(a,Pe))}
function rh(a,b,c){b=xf(b);if(b!=null){switch(typeof b){case "string":Eg(b)}if(b!=null)switch(Ig(a.h,c*8),typeof b){case "number":a=a.h;Se(b);Hg(a,Oe,Pe);break;case "bigint":c=BigInt.asUintN(64,b);c=new Dg(Number(c&BigInt(4294967295)),Number(c>>BigInt(32)));Hg(a.h,c.i,c.h);break;default:c=Eg(b),Hg(a.h,c.i,c.h)}}}
function sh(a,b,c){b=lf(b);if(b!=null&&b!=null)if(Ig(a.h,c*8),a=a.h,c=b,c>=0)Ig(a,c);else{for(b=0;b<9;b++)a.h.push(c&127|128),c>>=7;a.h.push(1)}}
function th(a,b,c){b=b==null||typeof b==="boolean"?b:typeof b==="number"?!!b:void 0;b!=null&&(Ig(a.h,c*8),a.h.h.push(b?1:0))}
function uh(a,b,c){b=Bf(b);b!=null&&Pg(a,c,dd(b))}
function vh(a,b,c,d,e){b=bh(b,d);b!=null&&(c=Mg(a,c),e(b,a),Ng(a,c))}
var wh=new Zg(qh,Xg),xh=new Zg(qh,Xg),yh=new Zg(rh,Vg),zh=new Zg(rh,Vg),Ah=new Zg(sh,Ug),Bh=new Zg(sh,Ug),Ch;Ch=new Zg(function(a,b,c){Og(a,c,yf(b))},Wg);
var Dh;Dh=new Zg(function(a,b,c){b=ph(yf,b,!1);if(b!=null)for(var d=0;d<b.length;d++)Og(a,c,b[d])},Wg);
var Eh=new Zg(th,Sg),Fh=new Zg(th,Sg),Gh=new Zg(uh,Tg),Hh;Hh=new Zg(function(a,b,c){b=ph(Bf,b,!0);if(b!=null)for(var d=0;d<b.length;d++){var e=a,f=c,g=b[d];g!=null&&Pg(e,f,dd(g))}},Tg);
var Ih=new Zg(uh,Tg),Jh,Kh=void 0;Kh=Kh===void 0?Rg:Kh;Jh=new Zg(function(a,b,c,d,e){if(Array.isArray(b)){for(var f=0;f<b.length;f++)vh(a,b[f],c,d,e);a=b[L]|0;a&1||oe(b,a|1)}},Kh);
var Lh=$g(vh);function Mh(){var a=Nh;this.ctor=Oh;this.isRepeated=0;this.h=qg;this.defaultValue=void 0;this.i=a.Pf!=null?xe:void 0}
Mh.prototype.register=function(){td(this)};function Ph(a){return function(b){return Yg(a,b)}}
;function Qh(a){this.D=M(a)}
v(Qh,N);function Rh(a,b){return jg(a,b,jf)}
;function Sh(a){this.D=M(a)}
v(Sh,N);var Th=[1,2,3];var Uh=[0,Th,Ih,Bh,Fh];var Vh=[0,Jh,[0,wh,yh]];function Wh(a){this.D=M(a)}
v(Wh,N);var Xh=[1,2,3];var Yh=[0,Xh,zh,xh,Lh,Vh];function Zh(a){this.D=M(a)}
v(Zh,N);var $h=[0,Jh,Uh,Yh];var ai=[0,Gh];function bi(a){this.D=M(a)}
v(bi,N);var ci=[0,Gh,-1,Eh];var di=[0,Gh,-1,Ah,Eh];function ei(a){this.D=M(a)}
v(ei,N);var fi=[1,2,3];var gi=[0,fi,Lh,ci,Lh,di,Lh,ai];function hi(a){this.D=M(a)}
v(hi,N);hi.prototype.j=function(a){return function(){var b=new Kg;nh(this.D,b,ih(a));Lg(b,b.h.end());for(var c=new Uint8Array(b.i),d=b.j,e=d.length,f=0,g=0;g<e;g++){var h=d[g];c.set(h,f);f+=h.length}b.j=[c];return c}}([0,
Gh,gi,Hh,Jh,$h,Ch,Dh]);function ii(a){this.D=M(a)}
v(ii,N);function ji(a){var b=new hi;b=yg(b,1,a.i);var c=ki(a);b=jg(b,c,zf);c=[];for(var d=[],e=y(a.h.keys()),f=e.next();!f.done;f=e.next())d.push(f.value.split(","));for(e=0;e<d.length;e++){f=d[e];for(var g=a.o,h=a.od(f)||[],k=[],l=0;l<h.length;l++){var m=h[l],n=m&&m.h;m=new Wh;switch(g){case 3:n=Number(n);Number.isFinite(n)&&mg(m,1,Xh,tf(n));break;case 2:n=Number(n);if(n!=null&&typeof n!=="number")throw Error("Value of float/double field must be a number, found "+typeof n+": "+n);mg(m,2,Xh,n)}k.push(m)}g=
k;for(h=0;h<g.length;h++){k=g[h];l=new Zh;k=sg(l,Wh,2,k);l=[];m=li(a);for(n=0;n<m.length;n++){var r=m[n],t=f[n],w=new Sh;switch(r){case 3:mg(w,1,Th,Af(String(t)));break;case 2:r=Number(t);Number.isFinite(r)&&mg(w,2,Th,kf(r));break;case 1:mg(w,3,Th,ff(t==="true"))}l.push(w)}tg(k,Sh,1,l);c.push(k)}}tg(b,Zh,4,c);return b}
;function mi(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a.indexOf("blob:")===0&&(a=a.substring(5));a=a.split("#")[0].split("?")[0];a=a.toLowerCase();a.indexOf("//")==0&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");c!=-1&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if(c!=="http"&&c!=="https"&&c!=="chrome-extension"&&
c!=="moz-extension"&&c!=="file"&&c!=="android-app"&&c!=="chrome-search"&&c!=="chrome-untrusted"&&c!=="chrome"&&c!=="app"&&c!=="devtools")throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(d!=-1){var e=b.substring(d+1);b=b.substring(0,d);if(c==="http"&&e!=="80"||c==="https"&&e!=="443")a=":"+e}return c+"://"+b+a}
;function ni(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;m=l=0}
function b(n){for(var r=g,t=0;t<64;t+=4)r[t/4]=n[t]<<24|n[t+1]<<16|n[t+2]<<8|n[t+3];for(t=16;t<80;t++)n=r[t-3]^r[t-8]^r[t-14]^r[t-16],r[t]=(n<<1|n>>>31)&4294967295;n=e[0];var w=e[1],z=e[2],x=e[3],H=e[4];for(t=0;t<80;t++){if(t<40)if(t<20){var F=x^w&(z^x);var V=1518500249}else F=w^z^x,V=1859775393;else t<60?(F=w&z|x&(w|z),V=2400959708):(F=w^z^x,V=3395469782);F=((n<<5|n>>>27)&4294967295)+F+H+V+r[t]&4294967295;H=x;x=z;z=(w<<30|w>>>2)&4294967295;w=n;n=F}e[0]=e[0]+n&4294967295;e[1]=e[1]+w&4294967295;e[2]=
e[2]+z&4294967295;e[3]=e[3]+x&4294967295;e[4]=e[4]+H&4294967295}
function c(n,r){if(typeof n==="string"){n=unescape(encodeURIComponent(n));for(var t=[],w=0,z=n.length;w<z;++w)t.push(n.charCodeAt(w));n=t}r||(r=n.length);t=0;if(l==0)for(;t+64<r;)b(n.slice(t,t+64)),t+=64,m+=64;for(;t<r;)if(f[l++]=n[t++],m++,l==64)for(l=0,b(f);t+64<r;)b(n.slice(t,t+64)),t+=64,m+=64}
function d(){var n=[],r=m*8;l<56?c(h,56-l):c(h,64-(l-56));for(var t=63;t>=56;t--)f[t]=r&255,r>>>=8;b(f);for(t=r=0;t<5;t++)for(var w=24;w>=0;w-=8)n[r++]=e[t]>>w&255;return n}
for(var e=[],f=[],g=[],h=[128],k=1;k<64;++k)h[k]=0;var l,m;a();return{reset:a,update:c,digest:d,Ze:function(){for(var n=d(),r="",t=0;t<n.length;t++)r+="0123456789ABCDEF".charAt(Math.floor(n[t]/16))+"0123456789ABCDEF".charAt(n[t]%16);return r}}}
;function oi(a,b,c){var d=String(D.location.href);return d&&a&&b?[b,pi(mi(d),a,c||null)].join(" "):null}
function pi(a,b,c){var d=[],e=[];if((Array.isArray(c)?2:1)==1)return e=[b,a],Tb(d,function(h){e.push(h)}),qi(e.join(" "));
var f=[],g=[];Tb(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=f.length==0?[c,b,a]:[f.join(":"),c,b,a];Tb(d,function(h){e.push(h)});
a=qi(e.join(" "));a=[c,a];g.length==0||a.push(g.join(""));return a.join("_")}
function qi(a){var b=ni();b.update(a);return b.Ze().toLowerCase()}
;function ri(a){this.h=a||{cookie:""}}
p=ri.prototype;p.isEnabled=function(){if(!D.navigator.cookieEnabled)return!1;if(this.h.cookie)return!0;this.set("TESTCOOKIESENABLED","1",{kc:60});if(this.get("TESTCOOKIESENABLED")!=="1")return!1;this.remove("TESTCOOKIESENABLED");return!0};
p.set=function(a,b,c){var d=!1;if(typeof c==="object"){var e=c.sameSite;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.kc}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');h===void 0&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=h<0?"":h==0?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+h*1E3)).toUTCString();this.h.cookie=a+"="+b+c+g+h+d+(e!=null?";samesite="+
e:"")};
p.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=jb(d[e]);if(f.lastIndexOf(c,0)==0)return f.slice(c.length);if(f==a)return""}return b};
p.remove=function(a,b,c){var d=this.get(a)!==void 0;this.set(a,"",{kc:0,path:b,domain:c});return d};
p.ec=function(){return si(this).keys};
p.bb=function(){return si(this).values};
p.clear=function(){for(var a=si(this).keys,b=a.length-1;b>=0;b--)this.remove(a[b])};
function si(a){a=(a.h.cookie||"").split(";");for(var b=[],c=[],d,e,f=0;f<a.length;f++)e=jb(a[f]),d=e.indexOf("="),d==-1?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));return{keys:b,values:c}}
var ti=new ri(typeof document=="undefined"?null:document);function ui(){var a=D.__SAPISID||D.__APISID||D.__3PSAPISID||D.__1PSAPISID||D.__OVERRIDE_SID;if(a)return!0;typeof document!=="undefined"&&(a=new ri(document),a=a.get("SAPISID")||a.get("APISID")||a.get("__Secure-3PAPISID")||a.get("__Secure-1PAPISID"));return!!a}
function vi(a,b,c,d){(a=D[a])||typeof document==="undefined"||(a=(new ri(document)).get(b));return a?oi(a,c,d):null}
function wi(a){var b=mi(D==null?void 0:D.location.href),c=[];if(ui()){b=b.indexOf("https:")==0||b.indexOf("chrome-extension:")==0||b.indexOf("chrome-untrusted://new-tab-page")==0||b.indexOf("moz-extension:")==0;var d=b?D.__SAPISID:D.__APISID;d||typeof document==="undefined"||(d=new ri(document),d=d.get(b?"SAPISID":"APISID")||d.get("__Secure-3PAPISID"));(d=d?oi(d,b?"SAPISIDHASH":"APISIDHASH",a):null)&&c.push(d);b&&((b=vi("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&c.push(b),(a=vi("__3PSAPISID",
"__Secure-3PAPISID","SAPISID3PHASH",a))&&c.push(a))}return c.length==0?null:c.join(" ")}
;function xi(){}
xi.prototype.compress=function(a){var b,c,d,e;return B(function(f){switch(f.h){case 1:return b=new CompressionStream("gzip"),c=(new Response(b.readable)).arrayBuffer(),d=b.writable.getWriter(),f.yield(d.write((new TextEncoder).encode(a)),2);case 2:return f.yield(d.close(),3);case 3:return e=Uint8Array,f.yield(c,4);case 4:return f.return(new e(f.i))}})};
xi.prototype.isSupported=function(a){return a<1024?!1:typeof CompressionStream!=="undefined"};function yi(a){this.D=M(a)}
v(yi,N);function zi(a,b){this.intervalMs=a;this.callback=b;this.enabled=!1;this.h=function(){return Za()};
this.i=this.h()}
zi.prototype.setInterval=function(a){this.intervalMs=a;this.timer&&this.enabled?(this.stop(),this.start()):this.timer&&this.stop()};
zi.prototype.start=function(){var a=this;this.enabled=!0;this.timer||(this.timer=setTimeout(function(){a.tick()},this.intervalMs),this.i=this.h())};
zi.prototype.stop=function(){this.enabled=!1;this.timer&&(clearTimeout(this.timer),this.timer=void 0)};
zi.prototype.tick=function(){var a=this;if(this.enabled){var b=Math.max(this.h()-this.i,0);b<this.intervalMs*.8?this.timer=setTimeout(function(){a.tick()},this.intervalMs-b):(this.timer&&(clearTimeout(this.timer),this.timer=void 0),this.callback(),this.enabled&&(this.stop(),this.start()))}else this.timer=void 0};function Ai(a){this.D=M(a)}
v(Ai,N);function Bi(a){this.D=M(a)}
v(Bi,N);function Ci(a,b){this.x=a!==void 0?a:0;this.y=b!==void 0?b:0}
p=Ci.prototype;p.clone=function(){return new Ci(this.x,this.y)};
p.equals=function(a){return a instanceof Ci&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
p.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
p.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
p.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};
p.scale=function(a,b){this.x*=a;this.y*=typeof b==="number"?b:a;return this};function Di(a,b){this.width=a;this.height=b}
p=Di.prototype;p.clone=function(){return new Di(this.width,this.height)};
p.aspectRatio=function(){return this.width/this.height};
p.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
p.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
p.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};
p.scale=function(a,b){this.width*=a;this.height*=typeof b==="number"?b:a;return this};function Ei(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function Fi(a){var b=[],c=0,d;for(d in a)b[c++]=a[d];return b}
function Gi(a){var b=Hi,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function Ii(a){for(var b in a)return!1;return!0}
function Ji(a,b){if(a!==null&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function Ki(a){return a!==null&&"privembed"in a?a.privembed:!1}
function Li(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function Mi(a){var b={},c;for(c in a)b[c]=a[c];return b}
function Ni(a){if(!a||typeof a!=="object")return a;if(typeof a.clone==="function")return a.clone();if(typeof Map!=="undefined"&&a instanceof Map)return new Map(a);if(typeof Set!=="undefined"&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:typeof ArrayBuffer!=="function"||typeof ArrayBuffer.isView!=="function"||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=Ni(a[c]);return b}
var Oi="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function Pi(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<Oi.length;f++)c=Oi[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;function Qi(a,b){this.h=a===Ri&&b||""}
Qi.prototype.toString=function(){return this.h};
var Ri={};new Qi(Ri,"");"ARTICLE SECTION NAV ASIDE H1 H2 H3 H4 H5 H6 HEADER FOOTER ADDRESS P HR PRE BLOCKQUOTE OL UL LH LI DL DT DD FIGURE FIGCAPTION MAIN DIV EM STRONG SMALL S CITE Q DFN ABBR RUBY RB RT RTC RP DATA TIME CODE VAR SAMP KBD SUB SUP I B U MARK BDI BDO SPAN BR WBR NOBR INS DEL PICTURE PARAM TRACK MAP TABLE CAPTION COLGROUP COL TBODY THEAD TFOOT TR TD TH SELECT DATALIST OPTGROUP OPTION OUTPUT PROGRESS METER FIELDSET LEGEND DETAILS SUMMARY MENU DIALOG SLOT CANVAS FONT CENTER ACRONYM BASEFONT BIG DIR HGROUP STRIKE TT".split(" ").concat(["BUTTON",
"INPUT"]);function Si(a){var b=document;return typeof a==="string"?b.getElementById(a):a}
function Ti(a){var b=document;a=String(a);b.contentType==="application/xhtml+xml"&&(a=a.toLowerCase());return b.createElement(a)}
function Ui(a){for(var b;b=a.firstChild;)a.removeChild(b)}
function Vi(a){a&&a.parentNode&&a.parentNode.removeChild(a)}
function Wi(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;function Xi(a){this.D=M(a)}
v(Xi,N);Xi.prototype.Hc=function(){return xg(this)};function Yi(a){this.D=M(a)}
v(Yi,N);function Zi(a){this.D=M(a)}
v(Zi,N);function $i(a){tg(aj,Yi,1,a)}
var bj=Ph(Zi);function cj(a){this.D=M(a)}
v(cj,N);var dj=["platform","platformVersion","architecture","model","uaFullVersion"],aj=new Zi,ej=null;function fj(a,b){b=b===void 0?dj:b;if(!ej){var c;a=(c=a.navigator)==null?void 0:c.userAgentData;if(!a||typeof a.getHighEntropyValues!=="function"||a.brands&&typeof a.brands.map!=="function")return Promise.reject(Error("UACH unavailable"));$i((a.brands||[]).map(function(e){var f=new Yi;f=yg(f,1,e.brand);return yg(f,2,e.version)}));
typeof a.mobile==="boolean"&&eg(aj,2,ff(a.mobile));ej=a.getHighEntropyValues(b)}var d=new Set(b);return ej.then(function(e){var f=aj.clone();d.has("platform")&&yg(f,3,e.platform);d.has("platformVersion")&&yg(f,4,e.platformVersion);d.has("architecture")&&yg(f,5,e.architecture);d.has("model")&&yg(f,6,e.model);d.has("uaFullVersion")&&yg(f,7,e.uaFullVersion);return f.serialize()}).catch(function(){return aj.serialize()})}
;function gj(a){this.D=M(a)}
v(gj,N);function hj(a){return zg(a,1,1)}
;function ij(a){this.D=M(a,4)}
v(ij,N);function jj(a){this.D=M(a,36)}
v(jj,N);function kj(a){this.D=M(a,19)}
v(kj,N);kj.prototype.sc=function(a){return zg(this,2,a)};function lj(a,b){this.Sa=b=b===void 0?!1:b;this.j=this.locale=null;this.i=0;this.isFinal=!1;this.h=new kj;Number.isInteger(a)&&this.h.sc(a);b||(this.locale=document.documentElement.getAttribute("lang"));mj(this,new gj)}
lj.prototype.sc=function(a){this.h.sc(a);return this};
function mj(a,b){sg(a.h,gj,1,b);xg(b)||hj(b);a.Sa||(b=nj(a),wg(b,5)||yg(b,5,a.locale));a.j&&(b=nj(a),qg(b,Zi,9)||sg(b,Zi,9,a.j))}
function oj(a,b){a.i=b}
function pj(a){var b=b===void 0?dj:b;var c=a.Sa?void 0:window;c?fj(c,b).then(function(d){a.j=bj(d!=null?d:"[]");d=nj(a);sg(d,Zi,9,a.j);return!0}).catch(function(){return!1}):Promise.resolve(!1)}
function nj(a){var b=qg(a.h,gj,1);b||(b=new gj,mj(a,b));a=b;b=qg(a,cj,11);b||(b=new cj,sg(a,cj,11,b));return b}
function qj(a,b,c,d,e,f,g){c=c===void 0?0:c;d=d===void 0?0:d;e=e===void 0?null:e;f=f===void 0?0:f;g=g===void 0?0:g;if(!a.Sa){var h=nj(a);var k=new Xi;k=zg(k,1,a.i);k=eg(k,2,ff(a.isFinal));d=eg(k,3,kf(d>0?d:void 0));d=eg(d,4,kf(f>0?f:void 0));d=eg(d,5,kf(g>0?g:void 0));f=d.D;g=f[L]|0;d=se(d,g)?d:Vf(d,f,g)?Wf(d,f):new d.constructor(Uf(f,g,!0));sg(h,Xi,10,d)}a=a.h.clone();h=Date.now().toString();a=eg(a,4,tf(h));b=b.slice();b=tg(a,jj,3,b);e&&(a=new Ai,e=eg(a,13,kf(e)),a=new Bi,e=sg(a,Ai,2,e),a=new ij,
e=sg(a,Bi,1,e),e=zg(e,2,9),sg(b,ij,18,e));c&&eg(b,14,tf(c));return b}
;var rj=function(){if(!D.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{var c=function(){};
D.addEventListener("test",c,b);D.removeEventListener("test",c,b)}catch(d){}return a}();function sj(a,b,c,d){this.o=a;this.u=b;this.h=this.j=a;this.H=c||0;this.B=d||2}
sj.prototype.i=0;sj.prototype.reset=function(){this.h=this.j=this.o;this.i=0};
sj.prototype.getValue=function(){return this.j};
function tj(a){a.h=Math.min(a.u,a.h*a.B);a.j=Math.min(a.u,a.h+(a.H?Math.round(a.H*(Math.random()-.5)*2*a.h):0));a.i++}
;function Nh(a){this.D=M(a,8)}
v(Nh,N);var uj=Ph(Nh);function Oh(a){this.D=M(a)}
v(Oh,N);var vj;vj=new Mh;function wj(a){J.call(this);var b=this;this.componentId="";this.h=[];this.Wa="";this.pageId=null;this.lb=this.na=-1;this.G=this.experimentIds=null;this.B=this.o=0;this.V=null;this.ba=this.ha=0;this.Ub=1;this.timeoutMillis=0;this.za=!1;this.logSource=a.logSource;this.Eb=a.Eb||function(){};
this.j=new lj(a.logSource,a.Sa);this.network=a.network||null;this.ub=a.ub||null;this.bufferSize=1E3;this.K=a.Gg||null;this.sessionIndex=a.sessionIndex||null;this.cc=a.cc||!1;this.logger=null;this.withCredentials=!a.kd;this.Sa=a.Sa||!1;this.aa=!this.Sa&&!!window&&!!window.navigator&&window.navigator.sendBeacon!==void 0;this.Va=typeof URLSearchParams!=="undefined"&&!!(new URL(xj())).searchParams&&!!(new URL(xj())).searchParams.set;var c=hj(new gj);mj(this.j,c);this.u=new sj(1E4,3E5,.1);a=yj(this,a.Ld);
this.i=new zi(this.u.getValue(),a);this.Ga=new zi(6E5,a);this.cc||this.Ga.start();this.Sa||(document.addEventListener("visibilitychange",function(){if(document.visibilityState==="hidden"){zj(b);var d;(d=b.V)==null||d.flush()}}),document.addEventListener("pagehide",function(){zj(b);
var d;(d=b.V)==null||d.flush()}))}
v(wj,J);function yj(a,b){return a.Va?b?function(){b().then(function(){a.flush()})}:function(){a.flush()}:function(){}}
wj.prototype.X=function(){zj(this);this.i.stop();this.Ga.stop();J.prototype.X.call(this)};
function Aj(a){a.K||(a.K=xj());try{return(new URL(a.K)).toString()}catch(b){return(new URL(a.K,window.location.origin)).toString()}}
function Bj(a,b,c){a.V&&a.V.Ia(b,c)}
wj.prototype.log=function(a){Bj(this,2,1);if(this.Va){a=a.clone();var b=this.Ub++;a=eg(a,21,tf(b));this.componentId&&yg(a,26,this.componentId);b=a;var c=cg(b,1);var d=d===void 0?!1:d;var e=typeof c;d=c==null?c:e==="bigint"?String($e(64,c)):hf(c)?e==="string"?of(c):d?pf(c):sf(c):void 0;d==null&&(d=Date.now(),d=Number.isFinite(d)?d.toString():"0",eg(b,1,tf(d)));(id?vf(cg(b,15,void 0,void 0,wf)):vf(cg(b,15)))==null&&eg(b,15,tf((new Date).getTimezoneOffset()*60));this.experimentIds&&(d=this.experimentIds.clone(),
sg(b,yi,16,d));Bj(this,1,1);b=this.h.length-this.bufferSize+1;b>0&&(this.h.splice(0,b),this.o+=b,Bj(this,3,b));this.h.push(a);this.cc||this.i.enabled||this.i.start()}};
wj.prototype.flush=function(a,b){var c=this;if(this.h.length===0)a&&a();else if(this.za&&this.aa)this.j.i=3,Cj(this);else{var d=Date.now();if(this.lb>d&&this.na<d)b&&b("throttled");else{this.network&&(typeof this.network.Hc==="function"?oj(this.j,this.network.Hc()):this.j.i=0);var e=this.h.length,f=qj(this.j,this.h,this.o,this.B,this.ub,this.ha,this.ba),g=this.Eb();if(g&&this.Wa===g)b&&b("stale-auth-token");else{this.h=[];this.i.enabled&&this.i.stop();this.o=0;d=f.serialize();var h;this.G&&this.G.isSupported(d.length)&&
(h=this.G.compress(d));var k=Dj(this,d,g),l=function(r){c.u.reset();c.i.setInterval(c.u.getValue());if(r){var t=null;try{var w=JSON.stringify(JSON.parse(r.replace(")]}'\n","")));t=uj(w)}catch(H){}if(t){r=Number(vg(t,1,He("-1")));r>0&&(c.na=Date.now(),c.lb=c.na+r);r=$a(ce);var z;Zd&&r&&((z=t.D[r])==null?void 0:z[175237375])!=null&&Yd(de,3);a:{var x=x===void 0?!1:x;if($a(je)&&$a(ce)&&void 0===je){z=t.D;r=z[ce];if(!r)break a;if(r=r.wj)try{r(z,175237375,Hf);break a}catch(H){ed(H)}}x&&(x=t.D,(z=$a(ce))&&
z in x&&(x=x[z])&&delete x[175237375])}x=vj.ctor?vj.h(t,vj.ctor,175237375,vj.i):vj.h(t,175237375,null,vj.i);if(x=x===null?void 0:x)x=Nc(x,1,-1),x!==-1&&(c.u=new sj(x<1?1:x,3E5,.1),c.i.setInterval(c.u.getValue()))}}a&&a();c.B=0},m=function(r,t){var w=Lc(f,jj,3);
var z=Number(vg(f,14));tj(c.u);c.i.setInterval(c.u.getValue());r===401&&g&&(c.Wa=g);z&&(c.o+=z);t===void 0&&(t=c.isRetryable(r));t&&(c.h=w.concat(c.h),c.cc||c.i.enabled||c.i.start());Bj(c,7,1);b&&b("net-send-failed",r);++c.B},n=function(){c.network&&c.network.send(k,l,m)};
h?h.then(function(r){Bj(c,5,e);k.Wc["Content-Encoding"]="gzip";k.Wc["Content-Type"]="application/binary";k.body=r;k.Re=2;n()},function(){Bj(c,6,e);
n()}):n()}}}};
function Dj(a,b,c){c=c===void 0?null:c;var d=d===void 0?a.withCredentials:d;var e={},f=new URL(Aj(a));c&&(e.Authorization=c);a.sessionIndex&&(e["X-Goog-AuthUser"]=a.sessionIndex,f.searchParams.set("authuser",a.sessionIndex));a.pageId&&(Object.defineProperty(e,"X-Goog-PageId",{value:a.pageId}),f.searchParams.set("pageId",a.pageId));return{url:f.toString(),body:b,Re:1,Wc:e,requestType:"POST",withCredentials:d,timeoutMillis:a.timeoutMillis}}
function zj(a){a.j.isFinal=!0;a.flush();a.j.isFinal=!1}
function Cj(a){Ej(a,function(b,c){b=new URL(b);b.searchParams.set("format","json");var d=!1;try{d=window.navigator.sendBeacon(b.toString(),c.serialize())}catch(e){}d||(a.aa=!1);return d})}
function Ej(a,b){if(a.h.length!==0){var c=new URL(Aj(a));c.searchParams.delete("format");var d=a.Eb();d&&c.searchParams.set("auth",d);c.searchParams.set("authuser",a.sessionIndex||"0");for(d=0;d<10&&a.h.length;++d){var e=a.h.slice(0,32),f=qj(a.j,e,a.o,a.B,a.ub,a.ha,a.ba);if(!b(c.toString(),f)){++a.B;break}a.o=0;a.B=0;a.ha=0;a.ba=0;a.h=a.h.slice(e.length)}a.i.enabled&&a.i.stop()}}
wj.prototype.isRetryable=function(a){return 500<=a&&a<600||a===401||a===0};
function xj(){return"https://play.google.com/log?format=json&hasfast=true"}
;function Fj(){this.Ke=typeof AbortController!=="undefined"}
Fj.prototype.send=function(a,b,c){var d=this,e,f,g,h,k,l,m,n,r,t;return B(function(w){switch(w.h){case 1:return f=(e=d.Ke?new AbortController:void 0)?setTimeout(function(){e.abort()},a.timeoutMillis):void 0,wa(w,2,3),g=Object.assign({},{method:a.requestType,
headers:Object.assign({},a.Wc)},a.body&&{body:a.body},a.withCredentials&&{credentials:"include"},{signal:a.timeoutMillis&&e?e.signal:null}),w.yield(fetch(a.url,g),5);case 5:h=w.i;if(h.status!==200){(k=c)==null||k(h.status);w.v(3);break}if((l=b)==null){w.v(7);break}return w.yield(h.text(),8);case 8:l(w.i);case 7:case 3:za(w);clearTimeout(f);Aa(w,0);break;case 2:m=ya(w);switch((n=m)==null?void 0:n.name){case "AbortError":(r=c)==null||r(408);break;default:(t=c)==null||t(400)}w.v(3)}})};
Fj.prototype.Hc=function(){return 4};function Gj(a,b){b=b===void 0?"0":b;J.call(this);this.logSource=a;this.sessionIndex=b;this.ab="https://play.google.com/log?format=json&hasfast=true";this.buildLabel=null;this.j=!1;this.network=null;this.componentId="";this.h=this.ub=null;this.i=!1;this.pageId=null;this.bufferSize=void 0}
v(Gj,J);function Hj(a,b){a.buildLabel=b;return a}
function Ij(a,b){a.network=b;return a}
function Jj(a,b){a.h=b}
function Kj(a){a.i=!0;return a}
Gj.prototype.kd=function(){this.o=!0;return this};
function Lj(a){a.network||(a.network=new Fj);var b=new wj({logSource:a.logSource,Eb:a.Eb?a.Eb:wi,sessionIndex:a.sessionIndex,Gg:a.ab,Sa:a.j,cc:!1,kd:a.o,Ld:a.Ld,network:a.network});Ac(a,b);if(a.buildLabel){var c=a.buildLabel,d=nj(b.j);yg(d,7,c)}b.G=new xi;a.componentId&&(b.componentId=a.componentId);a.ub&&(b.ub=a.ub);a.pageId&&(b.pageId=a.pageId);a.h&&((d=a.h)?(b.experimentIds||(b.experimentIds=new yi),c=b.experimentIds,d=d.serialize(),yg(c,4,d)):b.experimentIds&&eg(b.experimentIds,4));a.i&&(b.za=
b.aa);pj(b.j);a.bufferSize&&(b.bufferSize=a.bufferSize);a.network.sc&&a.network.sc(a.logSource);a.network.tg&&a.network.tg(b);return b}
;function Mj(a,b,c,d,e,f,g){a=a===void 0?-1:a;b=b===void 0?"":b;c=c===void 0?"":c;d=d===void 0?!1:d;e=e===void 0?"":e;J.call(this);this.logSource=a;this.componentId=b;f?b=f:(a=new Gj(a,"0"),a.componentId=b,Ac(this,a),c!==""&&(a.ab=c),d&&(a.j=!0),e&&Hj(a,e),g&&Ij(a,g),b=Lj(a));this.h=b}
v(Mj,J);Mj.prototype.flush=function(a){var b=a||[];if(b.length){a=new ii;for(var c=[],d=0;d<b.length;d++){var e=b[d],f=ji(e);c.push(f);e.clear()}tg(a,hi,1,c);b=this.h;if(a instanceof jj)b.log(a);else try{var g=new jj,h=a.serialize();var k=yg(g,8,h);b.log(k)}catch(l){Bj(b,4,1)}this.h.flush()}};function Nj(a){this.h=a}
;function Oj(a,b,c){this.i=a;this.o=b;this.fields=c||[];this.h=new Map}
function li(a){return a.fields.map(function(b){return b.fieldType})}
function ki(a){return a.fields.map(function(b){return b.fieldName})}
p=Oj.prototype;p.Le=function(a){var b=C.apply(1,arguments),c=this.od(b);c?c.push(new Nj(a)):this.qe(a,b)};
p.qe=function(a){var b=this.Kd(C.apply(1,arguments));this.h.set(b,[new Nj(a)])};
p.od=function(){var a=this.Kd(C.apply(0,arguments));return this.h.has(a)?this.h.get(a):void 0};
p.wf=function(){var a=this.od(C.apply(0,arguments));return a&&a.length?a[0]:void 0};
p.clear=function(){this.h.clear()};
p.Kd=function(){var a=C.apply(0,arguments);return a?a.join(","):"key"};function Pj(a,b){Oj.call(this,a,3,b)}
v(Pj,Oj);Pj.prototype.j=function(a){var b=C.apply(1,arguments),c=0,d=this.wf(b);d&&(c=d.h);this.qe(c+a,b)};function Qj(a,b){Oj.call(this,a,2,b)}
v(Qj,Oj);Qj.prototype.record=function(a){this.Le(a,C.apply(1,arguments))};function Rj(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
Rj.prototype.stopPropagation=function(){this.j=!0};
Rj.prototype.preventDefault=function(){this.defaultPrevented=!0};function Sj(a,b){Rj.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
ab(Sj,Rj);
Sj.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;b=a.relatedTarget;b||(c=="mouseover"?b=a.fromElement:c=="mouseout"&&(b=a.toElement));this.relatedTarget=b;d?(this.clientX=d.clientX!==void 0?d.clientX:d.pageX,this.clientY=d.clientY!==void 0?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||0):(this.clientX=a.clientX!==void 0?a.clientX:a.pageX,this.clientY=a.clientY!==
void 0?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||(c=="keypress"?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType=a.pointerType;this.state=a.state;this.i=a;a.defaultPrevented&&Sj.Ca.preventDefault.call(this)};
Sj.prototype.stopPropagation=function(){Sj.Ca.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
Sj.prototype.preventDefault=function(){Sj.Ca.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var Tj="closure_listenable_"+(Math.random()*1E6|0);var Uj=0;function Vj(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.handler=e;this.key=++Uj;this.qc=this.Bc=!1}
function Wj(a){a.qc=!0;a.listener=null;a.proxy=null;a.src=null;a.handler=null}
;function Xj(a){this.src=a;this.listeners={};this.h=0}
Xj.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=Yj(a,b,d,e);g>-1?(b=a[g],c||(b.Bc=!1)):(b=new Vj(b,this.src,f,!!d,e),b.Bc=c,a.push(b));return b};
Xj.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=Yj(e,b,c,d);return b>-1?(Wj(e[b]),Array.prototype.splice.call(e,b,1),e.length==0&&(delete this.listeners[a],this.h--),!0):!1};
function Zj(a,b){var c=b.type;c in a.listeners&&Yb(a.listeners[c],b)&&(Wj(b),a.listeners[c].length==0&&(delete a.listeners[c],a.h--))}
function Yj(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.qc&&f.listener==b&&f.capture==!!c&&f.handler==d)return e}return-1}
;var ak="closure_lm_"+(Math.random()*1E6|0),bk={},ck=0;function dk(a,b,c,d,e){if(d&&d.once)ek(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)dk(a,b[f],c,d,e);else c=fk(c),a&&a[Tj]?a.listen(b,c,Pa(d)?!!d.capture:!!d,e):gk(a,b,c,!1,d,e)}
function gk(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Pa(e)?!!e.capture:!!e,h=hk(a);h||(a[ak]=h=new Xj(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=ik();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)rj||(e=g),e===void 0&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(jk(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");ck++}}
function ik(){function a(c){return b.call(a.src,a.listener,c)}
var b=kk;return a}
function ek(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)ek(a,b[f],c,d,e);else c=fk(c),a&&a[Tj]?lk(a,b,c,Pa(d)?!!d.capture:!!d,e):gk(a,b,c,!0,d,e)}
function mk(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)mk(a,b[f],c,d,e);else(d=Pa(d)?!!d.capture:!!d,c=fk(c),a&&a[Tj])?a.i.remove(String(b),c,d,e):a&&(a=hk(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=Yj(b,c,d,e)),(c=a>-1?b[a]:null)&&nk(c))}
function nk(a){if(typeof a!=="number"&&a&&!a.qc){var b=a.src;if(b&&b[Tj])Zj(b.i,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(jk(c),d):b.addListener&&b.removeListener&&b.removeListener(d);ck--;(c=hk(b))?(Zj(c,a),c.h==0&&(c.src=null,b[ak]=null)):Wj(a)}}}
function jk(a){return a in bk?bk[a]:bk[a]="on"+a}
function kk(a,b){if(a.qc)a=!0;else{b=new Sj(b,this);var c=a.listener,d=a.handler||a.src;a.Bc&&nk(a);a=c.call(d,b)}return a}
function hk(a){a=a[ak];return a instanceof Xj?a:null}
var ok="__closure_events_fn_"+(Math.random()*1E9>>>0);function fk(a){if(typeof a==="function")return a;a[ok]||(a[ok]=function(b){return a.handleEvent(b)});
return a[ok]}
;function pk(){J.call(this);this.i=new Xj(this);this.za=this;this.ba=null}
ab(pk,J);pk.prototype[Tj]=!0;p=pk.prototype;p.addEventListener=function(a,b,c,d){dk(this,a,b,c,d)};
p.removeEventListener=function(a,b,c,d){mk(this,a,b,c,d)};
function qk(a,b){var c=a.ba;if(c){var d=[];for(var e=1;c;c=c.ba)d.push(c),++e}a=a.za;c=b.type||b;typeof b==="string"?b=new Rj(b,a):b instanceof Rj?b.target=b.target||a:(e=b,b=new Rj(c,a),Pi(b,e));e=!0;var f;if(d)for(f=d.length-1;!b.j&&f>=0;f--){var g=b.h=d[f];e=rk(g,c,!0,b)&&e}b.j||(g=b.h=a,e=rk(g,c,!0,b)&&e,b.j||(e=rk(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=rk(g,c,!1,b)&&e}
p.X=function(){pk.Ca.X.call(this);this.removeAllListeners();this.ba=null};
p.listen=function(a,b,c,d){return this.i.add(String(a),b,!1,c,d)};
function lk(a,b,c,d,e){a.i.add(String(b),c,!0,d,e)}
p.removeAllListeners=function(a){if(this.i){var b=this.i;a=a&&a.toString();var c=0,d;for(d in b.listeners)if(!a||d==a){for(var e=b.listeners[d],f=0;f<e.length;f++)++c,Wj(e[f]);delete b.listeners[d];b.h--}b=c}else b=0;return b};
function rk(a,b,c,d){b=a.i.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.qc&&g.capture==c){var h=g.listener,k=g.handler||g.src;g.Bc&&Zj(a.i,g);e=h.call(k,d)!==!1&&e}}return e&&!d.defaultPrevented}
;var sk=typeof AsyncContext!=="undefined"&&typeof AsyncContext.Snapshot==="function"?function(a){return a&&AsyncContext.Snapshot.wrap(a)}:function(a){return a};function tk(a,b){this.j=a;this.o=b;this.i=0;this.h=null}
tk.prototype.get=function(){if(this.i>0){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function uk(a,b){a.o(b);a.i<100&&(a.i++,b.next=a.h,a.h=b)}
;function vk(){this.i=this.h=null}
vk.prototype.add=function(a,b){var c=wk.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
vk.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var wk=new tk(function(){return new xk},function(a){return a.reset()});
function xk(){this.next=this.scope=this.h=null}
xk.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
xk.prototype.reset=function(){this.next=this.scope=this.h=null};var yk,zk=!1,Ak=new vk;function Bk(a,b){yk||Ck();zk||(yk(),zk=!0);Ak.add(a,b)}
function Ck(){var a=Promise.resolve(void 0);yk=function(){a.then(Dk)}}
function Dk(){for(var a;a=Ak.remove();){try{a.h.call(a.scope)}catch(b){ed(b)}uk(wk,a)}zk=!1}
;function Ek(){}
function Fk(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;function Gk(a){this.Z=0;this.ib=void 0;this.Bb=this.Ya=this.parent_=null;this.Ic=this.md=!1;if(a!=Ek)try{var b=this;a.call(void 0,function(c){Hk(b,2,c)},function(c){Hk(b,3,c)})}catch(c){Hk(this,3,c)}}
function Ik(){this.next=this.context=this.h=this.i=this.child=null;this.j=!1}
Ik.prototype.reset=function(){this.context=this.h=this.i=this.child=null;this.j=!1};
var Jk=new tk(function(){return new Ik},function(a){a.reset()});
function Kk(a,b,c){var d=Jk.get();d.i=a;d.h=b;d.context=c;return d}
function Lk(a){return new Gk(function(b,c){c(a)})}
Gk.prototype.then=function(a,b,c){return Mk(this,sk(typeof a==="function"?a:null),sk(typeof b==="function"?b:null),c)};
Gk.prototype.$goog_Thenable=!0;function Nk(a,b,c,d){Ok(a,Kk(b||Ek,c||null,d))}
p=Gk.prototype;p.finally=function(a){var b=this;a=sk(a);return new Gk(function(c,d){Nk(b,function(e){a();c(e)},function(e){a();
d(e)})})};
p.cd=function(a,b){return Mk(this,null,sk(a),b)};
p.catch=Gk.prototype.cd;p.cancel=function(a){if(this.Z==0){var b=new Pk(a);Bk(function(){Qk(this,b)},this)}};
function Qk(a,b){if(a.Z==0)if(a.parent_){var c=a.parent_;if(c.Ya){for(var d=0,e=null,f=null,g=c.Ya;g&&(g.j||(d++,g.child==a&&(e=g),!(e&&d>1)));g=g.next)e||(f=g);e&&(c.Z==0&&d==1?Qk(c,b):(f?(d=f,d.next==c.Bb&&(c.Bb=d),d.next=d.next.next):Rk(c),Sk(c,e,3,b)))}a.parent_=null}else Hk(a,3,b)}
function Ok(a,b){a.Ya||a.Z!=2&&a.Z!=3||Tk(a);a.Bb?a.Bb.next=b:a.Ya=b;a.Bb=b}
function Mk(a,b,c,d){var e=Kk(null,null,null);e.child=new Gk(function(f,g){e.i=b?function(h){try{var k=b.call(d,h);f(k)}catch(l){g(l)}}:f;
e.h=c?function(h){try{var k=c.call(d,h);k===void 0&&h instanceof Pk?g(h):f(k)}catch(l){g(l)}}:g});
e.child.parent_=a;Ok(a,e);return e.child}
p.Eg=function(a){this.Z=0;Hk(this,2,a)};
p.Fg=function(a){this.Z=0;Hk(this,3,a)};
function Hk(a,b,c){if(a.Z==0){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.Z=1;a:{var d=c,e=a.Eg,f=a.Fg;if(d instanceof Gk){Nk(d,e,f,a);var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(l){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Pa(d))try{var k=d.then;if(typeof k==="function"){Uk(d,k,e,f,a);g=!0;break a}}catch(l){f.call(a,l);g=!0;break a}g=!1}}}g||(a.ib=c,a.Z=b,a.parent_=null,Tk(a),b!=3||c instanceof Pk||Vk(a,c))}}
function Uk(a,b,c,d,e){function f(k){h||(h=!0,d.call(e,k))}
function g(k){h||(h=!0,c.call(e,k))}
var h=!1;try{b.call(a,g,f)}catch(k){f(k)}}
function Tk(a){a.md||(a.md=!0,Bk(a.lf,a))}
function Rk(a){var b=null;a.Ya&&(b=a.Ya,a.Ya=b.next,b.next=null);a.Ya||(a.Bb=null);return b}
p.lf=function(){for(var a;a=Rk(this);)Sk(this,a,this.Z,this.ib);this.md=!1};
function Sk(a,b,c,d){if(c==3&&b.h&&!b.j)for(;a&&a.Ic;a=a.parent_)a.Ic=!1;if(b.child)b.child.parent_=null,Wk(b,c,d);else try{b.j?b.i.call(b.context):Wk(b,c,d)}catch(e){Xk.call(null,e)}uk(Jk,b)}
function Wk(a,b,c){b==2?a.i.call(a.context,c):a.h&&a.h.call(a.context,c)}
function Vk(a,b){a.Ic=!0;Bk(function(){a.Ic&&Xk.call(null,b)})}
var Xk=ed;function Pk(a){ib.call(this,a)}
ab(Pk,ib);Pk.prototype.name="cancel";function Yk(a,b){pk.call(this);this.j=a||1;this.h=b||D;this.o=Xa(this.Bg,this);this.u=Za()}
ab(Yk,pk);p=Yk.prototype;p.enabled=!1;p.Fa=null;p.setInterval=function(a){this.j=a;this.Fa&&this.enabled?(this.stop(),this.start()):this.Fa&&this.stop()};
p.Bg=function(){if(this.enabled){var a=Za()-this.u;a>0&&a<this.j*.8?this.Fa=this.h.setTimeout(this.o,this.j-a):(this.Fa&&(this.h.clearTimeout(this.Fa),this.Fa=null),qk(this,"tick"),this.enabled&&(this.stop(),this.start()))}};
p.start=function(){this.enabled=!0;this.Fa||(this.Fa=this.h.setTimeout(this.o,this.j),this.u=Za())};
p.stop=function(){this.enabled=!1;this.Fa&&(this.h.clearTimeout(this.Fa),this.Fa=null)};
p.X=function(){Yk.Ca.X.call(this);this.stop();delete this.h};function Zk(a){J.call(this);this.G=a;this.j=0;this.o=100;this.u=!1;this.i=new Map;this.B=new Set;this.flushInterval=3E4;this.h=new Yk(this.flushInterval);this.h.listen("tick",this.bd,!1,this);Ac(this,this.h)}
v(Zk,J);p=Zk.prototype;p.sendIsolatedPayload=function(a){this.u=a;this.o=1};
function $k(a){a.h.enabled||a.h.start();a.j++;a.j>=a.o&&a.bd()}
p.bd=function(){var a=this.i.values();a=[].concat(A(a)).filter(function(b){return b.h.size});
a.length&&this.G.flush(a,this.u);al(a);this.j=0;this.h.enabled&&this.h.stop()};
p.Wb=function(a){var b=C.apply(1,arguments);this.i.has(a)||this.i.set(a,new Pj(a,b))};
p.hd=function(a){var b=C.apply(1,arguments);this.i.has(a)||this.i.set(a,new Qj(a,b))};
function bl(a,b){return a.B.has(b)?void 0:a.i.get(b)}
p.Tb=function(a){this.Je(a,1,C.apply(1,arguments))};
p.Je=function(a,b){var c=C.apply(2,arguments),d=bl(this,a);d&&d instanceof Pj&&(d.j(b,c),$k(this))};
p.record=function(a,b){var c=C.apply(2,arguments),d=bl(this,a);d&&d instanceof Qj&&(d.record(b,c),$k(this))};
function al(a){for(var b=0;b<a.length;b++)a[b].clear()}
;function cl(a){switch(a){case 200:return 0;case 400:return 3;case 401:return 16;case 403:return 7;case 404:return 5;case 409:return 10;case 412:return 9;case 429:return 8;case 499:return 1;case 500:return 2;case 501:return 12;case 503:return 14;case 504:return 4;default:return 2}}
function dl(a){switch(a){case 0:return"OK";case 1:return"CANCELLED";case 2:return"UNKNOWN";case 3:return"INVALID_ARGUMENT";case 4:return"DEADLINE_EXCEEDED";case 5:return"NOT_FOUND";case 6:return"ALREADY_EXISTS";case 7:return"PERMISSION_DENIED";case 16:return"UNAUTHENTICATED";case 8:return"RESOURCE_EXHAUSTED";case 9:return"FAILED_PRECONDITION";case 10:return"ABORTED";case 11:return"OUT_OF_RANGE";case 12:return"UNIMPLEMENTED";case 13:return"INTERNAL";case 14:return"UNAVAILABLE";case 15:return"DATA_LOSS";
default:return""}}
;function el(a,b,c){c=c===void 0?{}:c;b=Error.call(this,b);this.message=b.message;"stack"in b&&(this.stack=b.stack);this.code=a;this.metadata=c;this.name="RpcError";Object.setPrototypeOf(this,this.constructor.prototype)}
v(el,Error);el.prototype.toString=function(){var a="RpcError("+(dl(this.code)||String(this.code))+")";this.message&&(a+=": "+this.message);return a};function fl(){}
fl.prototype.serialize=function(a){var b=[];gl(this,a,b);return b.join("")};
function gl(a,b,c){if(b==null)c.push("null");else{if(typeof b=="object"){if(Array.isArray(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),gl(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],typeof f!="function"&&(c.push(e),hl(d,c),c.push(":"),gl(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":hl(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var il={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\v":"\\u000b"},jl=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function hl(a,b){b.push('"',a.replace(jl,function(c){var d=il[c];d||(d="\\u"+(c.charCodeAt(0)|65536).toString(16).slice(1),il[c]=d);return d}),'"')}
;function kl(){pk.call(this);this.headers=new Map;this.h=!1;this.P=null;this.o=this.aa="";this.j=this.V=this.B=this.K=!1;this.G=0;this.u=null;this.na="";this.ha=!1}
ab(kl,pk);var ll=/^https?$/i,ml=["POST","PUT"],nl=[];function ol(a,b,c,d,e,f,g){var h=new kl;nl.push(h);b&&h.listen("complete",b);lk(h,"ready",h.Ve);f&&(h.G=Math.max(0,f));g&&(h.ha=g);h.send(a,c,d,e)}
p=kl.prototype;p.Ve=function(){this.dispose();Yb(nl,this)};
p.send=function(a,b,c,d){if(this.P)throw Error("[goog.net.XhrIo] Object is active with another request="+this.aa+"; newUri="+a);b=b?b.toUpperCase():"GET";this.aa=a;this.o="";this.K=!1;this.h=!0;this.P=new XMLHttpRequest;this.P.onreadystatechange=sk(Xa(this.de,this));try{this.getStatus(),this.V=!0,this.P.open(b,String(a),!0),this.V=!1}catch(g){this.getStatus();pl(this,g);return}a=c||"";c=new Map(this.headers);if(d)if(Object.getPrototypeOf(d)===Object.prototype)for(var e in d)c.set(e,d[e]);else if(typeof d.keys===
"function"&&typeof d.get==="function"){e=y(d.keys());for(var f=e.next();!f.done;f=e.next())f=f.value,c.set(f,d.get(f))}else throw Error("Unknown input type for opt_headers: "+String(d));d=Array.from(c.keys()).find(function(g){return"content-type"==g.toLowerCase()});
e=D.FormData&&a instanceof D.FormData;!(Sb(ml,b)>=0)||d||e||c.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");b=y(c);for(d=b.next();!d.done;d=b.next())c=y(d.value),d=c.next().value,c=c.next().value,this.P.setRequestHeader(d,c);this.na&&(this.P.responseType=this.na);"withCredentials"in this.P&&this.P.withCredentials!==this.ha&&(this.P.withCredentials=this.ha);try{this.u&&(clearTimeout(this.u),this.u=null),this.G>0&&(this.getStatus(),this.u=setTimeout(this.Dg.bind(this),this.G)),
this.getStatus(),this.B=!0,this.P.send(a),this.B=!1}catch(g){this.getStatus(),pl(this,g)}};
p.Dg=function(){typeof La!="undefined"&&this.P&&(this.o="Timed out after "+this.G+"ms, aborting",this.getStatus(),qk(this,"timeout"),this.abort(8))};
function pl(a,b){a.h=!1;a.P&&(a.j=!0,a.P.abort(),a.j=!1);a.o=b;ql(a);rl(a)}
function ql(a){a.K||(a.K=!0,qk(a,"complete"),qk(a,"error"))}
p.abort=function(){this.P&&this.h&&(this.getStatus(),this.h=!1,this.j=!0,this.P.abort(),this.j=!1,qk(this,"complete"),qk(this,"abort"),rl(this))};
p.X=function(){this.P&&(this.h&&(this.h=!1,this.j=!0,this.P.abort(),this.j=!1),rl(this,!0));kl.Ca.X.call(this)};
p.de=function(){this.I||(this.V||this.B||this.j?sl(this):this.Uf())};
p.Uf=function(){sl(this)};
function sl(a){if(a.h&&typeof La!="undefined")if(a.B&&(a.P?a.P.readyState:0)==4)setTimeout(a.de.bind(a),0);else if(qk(a,"readystatechange"),a.isComplete()){a.getStatus();a.h=!1;try{if(tl(a))qk(a,"complete"),qk(a,"success");else{try{var b=(a.P?a.P.readyState:0)>2?a.P.statusText:""}catch(c){b=""}a.o=b+" ["+a.getStatus()+"]";ql(a)}}finally{rl(a)}}}
function rl(a,b){if(a.P){a.u&&(clearTimeout(a.u),a.u=null);var c=a.P;a.P=null;b||qk(a,"ready");try{c.onreadystatechange=null}catch(d){}}}
p.isActive=function(){return!!this.P};
p.isComplete=function(){return(this.P?this.P.readyState:0)==4};
function tl(a){var b=a.getStatus();a:switch(b){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var c=!0;break a;default:c=!1}if(!c){if(b=b===0)a=String(a.aa).match(jc)[1]||null,!a&&D.self&&D.self.location&&(a=D.self.location.protocol.slice(0,-1)),b=!ll.test(a?a.toLowerCase():"");c=b}return c}
p.getStatus=function(){try{return(this.P?this.P.readyState:0)>2?this.P.status:-1}catch(a){return-1}};
p.getLastError=function(){return typeof this.o==="string"?this.o:String(this.o)};function ul(){}
ul.prototype.send=function(a,b,c){b=b===void 0?function(){}:b;
c=c===void 0?function(){}:c;
ol(a.url,function(d){d=d.target;if(tl(d)){try{var e=d.P?d.P.responseText:""}catch(f){e=""}b(e)}else c(d.getStatus())},a.requestType,a.body,a.Wc,a.timeoutMillis,a.withCredentials)};
ul.prototype.Hc=function(){return 1};function vl(a,b){this.logger=a;this.event=b;this.startTime=wl()}
vl.prototype.done=function(){this.logger.qb(this.event,wl()-this.startTime)};
function xl(){Qc.apply(this,arguments)}
v(xl,Qc);function yl(a,b,c){var d=wl();b=b();a.qb(c,wl()-d);return b}
function zl(){xl.apply(this,arguments)}
v(zl,xl);p=zl.prototype;p.jc=function(){};
p.Ka=function(){};
p.qb=function(){};
p.wa=function(){};
p.Ua=function(){};
p.Nc=function(){};
p.Lc=function(){};
p.Mc=function(){};
function Al(a){xl.call(this);var b=this;this.logger=a;this.addOnDisposeCallback(function(){return void b.logger.dispose()})}
v(Al,xl);p=Al.prototype;p.update=function(a){this.logger.dispose();this.logger=a};
p.Ka=function(a){this.logger.Ka(a)};
p.qb=function(a,b){this.logger.qb(a,b)};
p.wa=function(a){this.logger.wa(a)};
p.Ua=function(){this.logger.Ua()};
p.Nc=function(a){this.logger.Nc(a)};
p.Lc=function(a){this.logger.Lc(a)};
p.Mc=function(a){this.logger.Mc(a)};
p.jc=function(a){this.logger.jc(a)};
function Bl(a,b,c,d){a=Kj(Ij(Hj(new Gj(1828,"0"),a),new ul)).kd();b.length&&Jj(a,Rh(new Qh,b));d!==void 0&&(a.ab=d);var e=new Mj(1828,"","",!1,"",Lj(a));Ac(e,a);var f=new Zk({flush:function(g){try{e.flush(g)}catch(h){c(h)}}});
f.addOnDisposeCallback(function(){setTimeout(function(){try{f.bd()}finally{e.dispose()}})});
f.o=1E5;f.flushInterval=3E4;f.h.setInterval(3E4);return f}
function Cl(a,b){J.call(this);var c=this;this.callback=a;this.i=b;this.h=-b;this.addOnDisposeCallback(function(){return void clearTimeout(c.timer)})}
v(Cl,J);function Dl(a){if(a.timer===void 0){var b=Math.max(0,a.h+a.i-wl());a.timer=setTimeout(function(){try{a.callback()}finally{a.h=wl(),a.timer=void 0}},b)}}
function El(a,b){xl.call(this);this.metrics=a;this.qa=b}
v(El,xl);p=El.prototype;p.jc=function(a){this.metrics.xg.record(a,this.qa)};
p.Ka=function(a){this.metrics.eventCount.Ia(a,this.qa)};
p.qb=function(a,b){this.metrics.kf.record(b,a,this.qa)};
p.wa=function(a){this.metrics.errorCount.Ia(a,this.qa)};
p.Nc=function(a){this.metrics.Jg.Ia(a,this.qa)};
p.Lc=function(a){this.metrics.Se.Ia(a,this.qa)};
p.Mc=function(a){this.metrics.Ig.Ia(a,this.qa)};
function Fl(a,b){b=b===void 0?[]:b;var c={qa:a.qa||"_",nd:a.nd||[],sd:a.sd|0,ab:a.ab,Qc:a.Qc||function(){},
Rb:a.Rb||function(f,g){return Bl(f,g,c.Qc,c.ab)}},d=c.Rb("52",c.nd.concat(b));
El.call(this,{xg:new Tc(d),errorCount:new Xc(d),eventCount:new Vc(d),kf:new Wc(d),Ij:new Uc(d),Jg:new Yc(d),Se:new Zc(d),Ig:new $c(d)},c.qa);var e=this;this.options=c;this.service=d;this.j=!a.Rb;this.h=new Cl(function(){return void e.service.bd()},c.sd);
this.addOnDisposeCallback(function(){e.h.dispose();e.j&&e.service.dispose()});
b.slice().sort(ac)}
v(Fl,El);Fl.prototype.Ua=function(){Dl(this.h)};
function wl(){var a,b,c;return(c=(a=globalThis.performance)==null?void 0:(b=a.now)==null?void 0:b.call(a))!=null?c:Date.now()}
;function Gl(a){this.D=M(a)}
v(Gl,N);function Hl(a){this.D=M(a)}
v(Hl,N);function Il(a){this.D=M(a,0,"bfkj")}
v(Il,N);var Jl=function(a){return ye(function(b){return b instanceof a&&!se(b)})}(Il);
Il.Pf="bfkj";function Mc(a){this.D=M(a)}
v(Mc,N);function Kl(a){this.D=M(a)}
v(Kl,N);var Ll=Ph(Kl);function Ml(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function Nl(a,b){if(a.disable)return new zl;b=b?Kc(b):[];a={qa:a.qa,nd:a.nf,sd:a.Qf,ab:a.ab,Qc:a.Qc,Rb:a.Rb};b=b===void 0?[]:b;return new Fl(a,b)}
function Ol(a){function b(w,z,x,H){Promise.resolve().then(function(){k.done();h.Ua();h.dispose();g.resolve({Ne:w,wg:z,Yf:x,Te:H})})}
function c(w,z,x,H){if(!d.logger.I){var F="k";z?F="h":x&&(F="u");F!=="k"?H!==0&&(d.logger.Ka(F),d.logger.qb(F,w)):d.j<=0?(d.logger.Ka(F),d.logger.qb(F,w),d.j=Math.floor(Math.random()*200)):d.j--}}
J.call(this);var d=this;this.j=Math.floor(Math.random()*200);this.h=new Kl;if("challenge"in a&&Jl(a.challenge)){var e=wg(a.challenge,4,void 0,xe);var f=wg(a.challenge,5,void 0,xe);wg(a.challenge,7,void 0,xe)&&(this.h=Ll(wg(a.challenge,7,void 0,xe)))}else e=a.program,f=a.globalName;this.addOnDisposeCallback(function(){var w,z,x;return B(function(H){if(H.h==1)return H.yield(d.i,2);w=H.i;z=w.wg;(x=z)==null||x();H.h=0})});
this.logger=Nl(a.Lb||{},this.h);Ac(this,this.logger);var g=new Ml;this.i=g.promise;this.logger.Ka("t");var h=this.logger.share(),k=new vl(h,"t");if(!D[f])throw this.logger.wa(25),Error("EGOU");if(!D[f].a)throw this.logger.wa(26),Error("ELIU");try{var l=D[f].a;f=[];for(var m=[],n=Kc(this.h),r=0;r<n.length;r++)f.push(n[r]),m.push(1);var t=Oc(this.h);for(n=0;n<t.length;n++)f.push(t[n]),m.push(2);this.o=y(l(e,b,!0,a.Ae,c,[f,m],wg(this.h,5),!1)).next().value;this.Pb=g.promise.then(function(){})}catch(w){throw this.logger.wa(28),
w;
}}
v(Ol,J);p=Ol.prototype;p.snapshot=function(a){if(this.I)throw Error("Already disposed");this.logger.Ka("n");var b=this.logger.share();return this.i.then(function(c){var d=c.Ne;return new Promise(function(e){var f=new vl(b,"n");d(function(g){f.done();b.jc(g.length);b.Ua();b.dispose();e(g)},[a.Ha,
a.Zc,a.Ee,a.Ed])})})};
p.se=function(a){var b=this;if(this.I)throw Error("Already disposed");this.logger.Ka("n");var c=yl(this.logger,function(){return b.o([a.Ha,a.Zc,a.Ee,a.Ed])},"n");
this.logger.jc(c.length);this.logger.Ua();return c};
p.lc=function(a){this.i.then(function(b){var c;(c=b.Yf)==null||c(a)})};
p.Dc=function(a,b){return this.i.then(function(c){var d;return(d=c.Te)==null?void 0:d(a,b,!1)})};
p.uc=function(){return this.logger.share()};function Pl(a){if(!a)return null;a=Bf(cg(a,4,void 0,bg));return a===null||a===void 0?null:pb(a)}
;function Ql(){this.promises={};this.h=null}
function Rl(){Ql.instance||(Ql.instance=new Ql);return Ql.instance}
function Sl(a,b){return Tl(a,qg(b,Gl,1,xe),qg(b,Hl,2,xe),wg(b,3,void 0,xe))}
function Tl(a,b,c,d){if(!b&&!c)return Promise.resolve();if(!d)return Ul(b,c);var e;(e=a.promises)[d]||(e[d]=new Promise(function(f,g){Ul(b,c).then(function(){a.h=d;f()},function(h){delete a.promises[d];
g(h)})}));
return a.promises[d]}
function Ul(a,b){return b?Vl(b):a?Wl(a):Promise.resolve()}
function Vl(a){return new Promise(function(b,c){var d=Ti("SCRIPT"),e=Pl(a);Lb(d,e);d.onload=function(){Vi(d);b()};
d.onerror=function(){Vi(d);c(Error("EWLS"))};
(document.getElementsByTagName("HEAD")[0]||document.documentElement).appendChild(d)})}
function Wl(a){return new Promise(function(b){var c=Ti("SCRIPT");if(a){var d=Bf(cg(a,6,void 0,bg));d=d===null||d===void 0?null:Ib(d)}else d=null;c.textContent=Jb(d);Kb(c);(document.getElementsByTagName("HEAD")[0]||document.documentElement).appendChild(c);Vi(c);b()})}
;function Xl(a){this.D=M(a)}
v(Xl,N);function Yl(a,b){return lg(a,1,Af(b))}
function Zl(a,b){return lg(a,2,Af(b))}
;function $l(a){J.call(this);var b=this;this.options=a;this.B=new Ml;this.Pb=this.B.promise;this.u=new Ml;this.K=1;this.j=new Ml;this.o=[];this.isPaused=!1;this.Rc=a.Rc||function(){};
this.logger=new Al(Nl(a.Lb||{}));am(this,a.Oa,a.Af,a.xj,a.zj,Object.assign({},bm,a.Sb||{}));this.addOnDisposeCallback(function(){return void cm(b)})}
v($l,J);p=$l.prototype;p.snapshot=function(a){var b=this;return B(function(c){switch(c.h){case 1:if(b.I)throw Error("Already disposed");if(b.i||b.G){c.v(2);break}return c.yield(b.u.promise,2);case 2:if(!b.i){c.v(4);break}return c.yield(b.i.snapshot(a),5);case 5:return c.return(c.i);case 4:throw b.G;}})};
p.pause=function(){this.I||this.isPaused||(this.isPaused=!0,this.h&&this.h.pause())};
p.resume=function(){!this.I&&this.isPaused&&(this.isPaused=!1,this.h&&this.h.resume())};
p.checkForRefresh=function(){var a=this;return B(function(b){if(a.I)throw Error("Already disposed");var c;if(c=a.h)c=a.h,c.isExpired()?(dm(c),c.Xc(0),c=!0):c=!1,c=!c;return c?b.v(0):b.yield(a.j.promise,0)})};
function em(a){var b;return B(function(c){if(a.I)throw Error("Already disposed");(b=a.h)==null||fm(b);return c.yield(a.j.promise,0)})}
function cm(a){a.G=Error("Cancelled by dispose");a.u.resolve();Pc(a.B.promise);a.B.reject(Error("Cancelled by dispose"));a.logger.dispose();Promise.all(a.o).then(function(){var c;return B(function(d){(c=a.i)==null||c.dispose();a.i=void 0;d.h=0})});
a.o=[];var b;(b=a.h)==null||fm(b);Pc(a.j.promise);a.j.reject(Error("Cancelled by dispose"))}
p.lc=function(a){var b,c;(b=this.i)==null||(c=b.lc)==null||c.call(b,a)};
p.Dc=function(a,b){var c,d,e;return(e=(c=this.i)==null?void 0:(d=c.Dc)==null?void 0:d.call(c,a,b))!=null?e:Promise.resolve()};
function gm(a,b){var c=a.Rc;a.Rc=function(){c();b()}}
function hm(a,b){a.I||(a.i=b,a.logger.update(b.uc()),a.u.resolve(),a.B.resolve(void 0),a.Rc())}
p.handleError=function(a){if(!this.I){this.G=a;this.u.resolve();var b,c;(c=(b=this.options).Pc)==null||c.call(b,a)}};
function im(a,b){b&&(Promise.all(a.o).then(function(){return void b.dispose()}),a.o=[])}
function jm(a,b){a.K=b;var c,d;(d=(c=a.options).rj)==null||d.call(c,b)}
function km(a){a.I||(a.j.resolve(),a.j=new Ml)}
function am(a,b,c,d,e,f){d=d===void 0?Rl():d;e=e===void 0?Promise.resolve(void 0):e;var g,h,k,l,m,n,r,t,w,z,x,H;B(function(F){switch(F.h){case 1:return F.yield(0,3);case 3:h=null;if(!g){F.v(6);break}jm(a,7);wa(F,7);return F.yield(lm(g.snapshot({}),f.Ye,function(){return Promise.resolve("E:CTO")}),9);
case 9:h=F.i;xa(F,6);break;case 7:ya(F),h="E:UCE";case 6:k=void 0,l=g?f.ff:f.gf,m=new sj(l,f.hf,f.jf,f.ef),n=1;case 10:if(!(n<=f.maxAttempts)){F.v(12);break}if(n===1){F.v(13);break}jm(a,0);a.h=new mm(m.getValue(),f.dd,f.xe);return F.yield(a.h.promise,14);case 14:r=F.i,a.h=void 0,r===1?(n=1,m.reset()):tj(m);case 13:wa(F,15);t=void 0;if(c){t=c;F.v(17);break}jm(a,5);w=d.h;return F.yield(lm(nm(b,w,h),f.rf,function(){return Promise.reject(Error("RGF:Fetch timed out"))}),18);
case 18:t=F.i;case 17:return jm(a,3),F.yield(lm(Sl(d,t),f.Kf,function(){return Promise.reject(Error("DTZ:Script timed out"))}),19);
case 19:return jm(a,8),F.yield(e,20);case 20:return z=new Ol({challenge:t,Lb:a.options.Lb,Ae:a.options.Ae}),F.yield(lm(z.Pb,f.vg,function(){return Promise.reject(Error("QEG:Setup timed out"))}),21);
case 21:k=z;F.v(12);break;case 15:x=ya(F),a.handleError(x),km(a);case 11:n++;F.v(10);break;case 12:if(a.I){F.v(5);break}k&&(c=void 0,im(a,g),g=k,hm(a,k),km(a));jm(a,2);a.h=new mm(f.ke,f.dd,f.xe);a.isPaused&&a.h.pause();return F.yield(a.h.promise,22);case 22:a.h=void 0;if(a.I){F.v(5);break}F.v(3);break;case 5:(H=g)==null||H.dispose(),F.h=0}})}
p.uc=function(){return this.logger.share()};
var bm={ke:432E5,dd:3E5,xe:10,Ye:1E4,rf:3E4,Kf:3E4,vg:6E4,gf:1E3,ff:6E4,hf:6E5,jf:.25,ef:2,maxAttempts:10};function lm(a,b,c){var d,e=new Promise(function(f){d=setTimeout(f,b)});
return Promise.race([a.finally(function(){return void clearTimeout(d)}),
e.then(c)])}
function mm(a,b,c){var d=this;this.endTimeMs=0;this.h=null;this.isPaused=!1;this.tick=function(){if(!d.isPaused){var e=d.endTimeMs-Date.now();e<=d.i?(d.h=null,d.Xc(0)):d.h=setTimeout(d.tick,Math.min(e,d.dd))}};
this.dd=b;this.i=c;this.promise=new Promise(function(e){d.Xc=e});
om(this,a)}
function om(a,b){a.endTimeMs=Date.now()+b;a.tick()}
mm.prototype.pause=function(){this.isPaused||(this.isPaused=!0,dm(this))};
mm.prototype.resume=function(){this.isPaused&&(this.isPaused=!1,this.tick())};
function fm(a){dm(a);a.endTimeMs=0;a.isPaused=!1;a.Xc(1)}
function dm(a){a.h&&(clearTimeout(a.h),a.h=null)}
mm.prototype.isExpired=function(){return Date.now()>this.endTimeMs};function pm(a,b){try{return globalThis.sessionStorage.setItem(a,b),!0}catch(c){return!1}}
var qm,rm=(qm=Math.imul)!=null?qm:function(a,b){return a*b|0};
function sm(a,b,c,d){b=b===void 0?0:b;c=c===void 0?a.length:c;var e=0;for(d&&(e=sm(d));b<c;b++)d=typeof a==="string"?a.charCodeAt(b):a[b],e=rm(31,e)+d|0;return e}
function tm(a,b){return[sm(a,0,a.length>>1,b),sm(a,a.length>>1)]}
var um=[196,200,224,18];function wm(a){var b=y(tm(a,um));a=b.next().value;b=b.next().value;return a.toString(16)+b.toString(16)}
function xm(a,b){var c=tm(b);a=new Uint32Array(a.buffer);b=a[0];var d=y(c);c=d.next().value;d=d.next().value;for(var e=1;e<a.length;e+=2){for(var f=b,g=e,h=c,k=d,l=0;l<22;l++)g=g>>>8|g<<24,g+=f|0,g^=h+38293,f=f<<3|f>>>29,f^=g,k=k>>>8|k<<24,k+=h|0,k^=l+38293,h=h<<3|h>>>29,h^=k;f=[f,g];a[e]^=f[0];e+1<a.length&&(a[e+1]^=f[1])}}
function ym(a,b,c,d,e){var f=(4-(um.length+c.length)%4)%4,g=new Uint8Array(4+f+um.length+4+c.length),h=new DataView(g.buffer),k=0;h.setUint32(k,Math.random()*4294967295);k=k+4+f;g.set(um,k);k+=um.length;h.setUint32(k,e);g.set(c,k+4);xm(g,d);return a.la(b,function(l){return void globalThis.sessionStorage.removeItem(l)})?pm(b,Fd(g))?"s":"t":"i"}
function zm(a,b){var c=globalThis.sessionStorage.getItem(a);if(!c)return["m"];try{var d=Hd(c);xm(d,b)}catch(e){return globalThis.sessionStorage.removeItem(a),["c"]}for(b=4;b<7&&d[b]===0;)b++;for(c=0;c<um.length;c++)if(d[b++]!==um[c])return globalThis.sessionStorage.removeItem(a),["d"];c=(new DataView(d.buffer)).getUint32(b);return Math.floor(Date.now()/1E3)>=c?(globalThis.sessionStorage.removeItem(a),["e"]):["a",new Uint8Array(d.buffer,b+4)]}
function Am(a,b,c){c=c===void 0?[]:c;this.maxItems=a;this.h=b===void 0?0:b;this.i=c}
function Bm(a){var b=globalThis.sessionStorage.getItem("iU5q-!O9@$");if(!b)return new Am(a);var c=b.split(",");if(c.length<2)return globalThis.sessionStorage.removeItem("iU5q-!O9@$"),new Am(a);b=c.slice(1);b.length===1&&b[0]===""&&(b=[]);c=Number(c[0]);return isNaN(c)||c<0||c>b.length?(globalThis.sessionStorage.removeItem("iU5q-!O9@$"),new Am(a)):new Am(a,c,b)}
Am.prototype.serialize=function(){return String(this.h)+","+this.i.join()};
Am.prototype.la=function(a,b){var c=void 0;if(this.i[this.h]!==a){var d=this.i.indexOf(a);d!==-1?(this.i.splice(d,1),d<this.h&&this.h--,this.i.splice(this.h,0,a)):(c=this.i[this.h],this.i[this.h]=a)}this.h=(this.h+1)%this.maxItems;a=pm("iU5q-!O9@$",this.serialize());c&&a&&b(c);return a};
function Hc(a,b){this.logger=b;try{var c=globalThis.sessionStorage&&!!globalThis.sessionStorage.getItem&&!!globalThis.sessionStorage.setItem&&!!globalThis.sessionStorage.removeItem}catch(d){c=!1}c&&(this.index=Bm(a))}
function Cm(a,b,c,d,e){var f=a.index?yl(a.logger,function(){return ym(a.index,wm(b),c,d,e)},"W"):"u";
a.logger.Mc(f)}
function Dm(a,b,c){var d=y(a.index?yl(a.logger,function(){return zm(wm(b),c)},"R"):["u"]),e=d.next().value;
d=d.next().value;a.logger.Lc(e);return d}
;var Em={toString:function(a){var b=[],c=0;a-=-2147483648;b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(a%52);for(a=Math.floor(a/52);a>0;)b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".charAt(a%62),a=Math.floor(a/62);return b.join("")}};function Fm(a){function b(){c-=d;c-=e;c^=e>>>13;d-=e;d-=c;d^=c<<8;e-=c;e-=d;e^=d>>>13;c-=d;c-=e;c^=e>>>12;d-=e;d-=c;d^=c<<16;e-=c;e-=d;e^=d>>>5;c-=d;c-=e;c^=e>>>3;d-=e;d-=c;d^=c<<10;e-=c;e-=d;e^=d>>>15}
a=Gm(a);for(var c=2654435769,d=2654435769,e=314159265,f=a.length,g=f,h=0;g>=12;g-=12,h+=12)c+=Hm(a,h),d+=Hm(a,h+4),e+=Hm(a,h+8),b();e+=f;switch(g){case 11:e+=a[h+10]<<24;case 10:e+=a[h+9]<<16;case 9:e+=a[h+8]<<8;case 8:d+=a[h+7]<<24;case 7:d+=a[h+6]<<16;case 6:d+=a[h+5]<<8;case 5:d+=a[h+4];case 4:c+=a[h+3]<<24;case 3:c+=a[h+2]<<16;case 2:c+=a[h+1]<<8;case 1:c+=a[h+0]}b();return Em.toString(e)}
function Gm(a){for(var b=[],c=0;c<a.length;c++)b.push(a.charCodeAt(c));return b}
function Hm(a,b){return a[b+0]+(a[b+1]<<8)+(a[b+2]<<16)+(a[b+3]<<24)}
;function Im(a){J.call(this);this.logger=a;this.j=new Ml}
v(Im,J);function Jm(a,b){var c=setTimeout(function(){a.j.resolve()},b);
a.addOnDisposeCallback(function(){return void clearTimeout(c)})}
Im.prototype.Oc=function(a,b){var c=this.gb(a);b==null||b(c);return yl(this.logger,function(){return Fd(c,2)},this.i)};
function Km(a,b,c,d){return yl(a.logger,function(){return c?a.Oc(b,d):a.gb(b,d)},a.h)}
function Lm(a,b,c,d){Im.call(this,a);this.o=b;this.B=c;this.h="m";this.i="x";this.u=0;Jm(this,d)}
v(Lm,Im);Lm.prototype.gb=function(a,b){var c=this;this.logger.Ka(this.h);++this.u>=this.B&&this.j.resolve();var d=a();a=yl(this.logger,function(){return c.o(d)},"C");
if(a===void 0)throw new I(17,"YNJ:Undefined");if(!(a instanceof Uint8Array))throw new I(18,"ODM:Invalid");b==null||b(a);return a};
function Mm(a,b,c){Im.call(this,a);this.o=b;this.h="f";this.i="z";Jm(this,c)}
v(Mm,Im);Mm.prototype.gb=function(){return this.o};
function Nm(a,b,c){Im.call(this,a);this.o=b;this.h="w";this.i="z";Jm(this,c)}
v(Nm,Im);Nm.prototype.gb=function(){var a=this;return yl(this.logger,function(){return Hd(a.o)},"d")};
Nm.prototype.Oc=function(){return this.o};
function Om(a,b){Im.call(this,a);this.error=b;this.h="e";this.i="y"}
v(Om,Im);function Pm(a,b){var c=(b(a.error.message)+":"+b(a.error.stack)).substring(0,2048);b=c.length+1;c=Qm(c);var d=new Uint8Array(4+c.length);d.set([42,b&127|128,b>>7,a.error.code]);d.set(c,4);return d}
Om.prototype.gb=function(){if(this.o)return this.o;this.o=Pm(this,function(a){return"_"+Fm(a)});
return Pm(this,function(a){return a})};
function Rm(a,b,c){Im.call(this,a);this.o=b;this.clientState=c;this.h="S";this.i="q"}
v(Rm,Im);Rm.prototype.gb=function(){var a=Math.floor(Date.now()/1E3),b=[Math.random()*255,Math.random()*255],c=b.concat([this.o&255,this.clientState],[a>>24&255,a>>16&255,a>>8&255,a&255]);a=new Uint8Array(2+c.length);a[0]=34;a[1]=c.length;a.set(c,2);c=a.subarray(2);for(var d=b=b.length;d<c.length;++d)c[d]^=c[d%b];this.logger.Nc(this.clientState);return a};
function Qm(a){return globalThis.TextEncoder?(new TextEncoder).encode(a):fd(a)}
;var Sm={sf:3E4,yg:2E4};function Tm(a){J.call(this);var b=this;this.Ob=new Ml;this.j=0;this.i=void 0;this.state=2;this.vm=a.vm;this.Oa=a.Oa;this.Sb=Object.assign({},Sm,a.Sb||{});this.logger=a.vm.uc();var c;this.onError=(c=a.onError)!=null?c:function(){};
this.Jd=a.Jd||!1;if(Um(a)){var d=this.vm;this.o=function(){return em(d).catch(function(g){g=Ic(b,new I(b.h?20:32,"TRG:Disposed",g));b.i=g;var h;(h=b.h)==null||h.dispose();b.h=void 0;b.Ob.reject(g)})};
gm(d,function(){return void Vm(b)});
d.K===2&&Vm(this)}else this.o=a.qj,Vm(this);var e=this.logger.share();e.Ka("o");var f=new vl(e,"o");this.Ob.promise.then(function(){f.done();e.Ua();e.dispose()},function(){return void e.dispose()});
this.addOnDisposeCallback(function(){b.h?(b.h.dispose(),b.h=void 0):b.i?b.logger.Ua():(b.i=Ic(b,new I(32,"TNP:Disposed")),b.logger.Ua(),b.Ob.reject(b.i))});
Ac(this,this.logger)}
v(Tm,J);function Wm(a,b){if(!(b instanceof I))if(b instanceof el){var c=Error(b.toString());c.stack=b.stack;b=new I(11,"EBH:Error",c)}else b=new I(12,"BSO:Unknown",b);return Ic(a,b)}
function Vm(a){var b,c,d,e,f,g,h,k,l,m,n,r,t,w,z;return B(function(x){switch(x.h){case 1:b=void 0;a.j++;c=new Ml;a.vm instanceof $l&&a.vm.o.push(c.promise);if(!a.Jd){x.v(2);break}d=new Ml;setTimeout(function(){return void d.resolve()});
return x.yield(d.promise,2);case 2:return e=a.logger.share(),wa(x,4,5),a.state=5,f={},g=[],x.yield(lm(a.vm.snapshot({Ha:f,Ee:g}),a.Sb.yg,function(){return Promise.reject(new I(15,"MDA:Timeout"))}),7);
case 7:h=x.i;if(a.I)throw new I(a.h?20:32,"MDA:Disposed");k=g[0];a.state=6;return x.yield(lm(Xm(a.Oa,h),a.Sb.sf,function(){return Promise.reject(new I(10,"BWB:Timeout"))}),8);
case 8:l=x.i;if(a.I)throw new I(a.h?20:32,"BWB:Disposed");a.state=7;b=yl(e,function(){var F=Ym(a,l,c,k);F.j.promise.then(function(){return void a.o()});
return F},"i");
case 5:za(x);e.dispose();Aa(x,6);break;case 4:m=ya(x);(n=b)==null||n.dispose();if(!a.i){r=Wm(a,m);c.resolve();var H;if(H=a.vm instanceof $l&&a.j<2)a:if(m instanceof I)H=m.code!==32&&m.code!==20&&m.code!==10;else{if(m instanceof el)switch(m.code){case 2:case 13:case 14:case 4:break;default:H=!1;break a}H=!0}if(H)return t=(1+Math.random()*.25)*(a.h?6E4:1E3),w=setTimeout(function(){return void a.o()},t),a.addOnDisposeCallback(function(){return void clearTimeout(w)}),x.return();
a.i=r}e.wa(a.h?13:14);a.Ob.reject(a.i);return x.return();case 6:a.state=8,a.j=0,(z=a.h)==null||z.dispose(),a.h=b,a.Ob.resolve(),x.h=0}})}
function Ym(a,b,c,d){var e=ug(b,2)*1E3;if(e<=0)throw new I(31,"TTM:Invalid");if(wg(b,4))return new Nm(a.logger,wg(b,4),e);if(!ug(b,3))return new Mm(a.logger,Ud(ig(b)),e);if(!d)throw new I(4,"PMD:Undefined");d=d(Ud(ig(b)));if(!(d instanceof Function))throw new I(16,"APF:Failed");a.u=Math.floor((Date.now()+e)/1E3);a=new Lm(a.logger,d,ug(b,3),e);a.addOnDisposeCallback(function(){return void c.resolve()});
return a}
Tm.prototype.gb=function(a){return Zm(this,Object.assign({},a),!1)};
Tm.prototype.Oc=function(a){return Zm(this,Object.assign({},a),!0)};
function Ic(a,b){a.logger.wa(b.code);a.onError(b);return b}
function $m(a,b){b=b instanceof I?b:new I(5,"TVD:error",b);return Ic(a,b)}
function Zm(a,b,c){try{if(a.I)throw new I(21,"BNT:disposed");if(!a.h&&a.i)throw a.i;var d,e;return(e=(d=an(a,b,c))!=null?d:bn(a,b,c))!=null?e:cn(a,b,c)}catch(f){if(!b.Sf)throw $m(a,f);return dn(a,c,f)}}
function an(a,b,c){var d;return(d=a.h)==null?void 0:Km(d,function(){return en(a,b)},c,function(e){var f;
if(a.h instanceof Lm&&((f=b.Ac)==null?0:f.zg))try{var g;(g=a.cache)==null||Cm(g,en(a,b),e,b.Ac.Od,a.u-120)}catch(h){Ic(a,new I(24,"ELX:write",h))}})}
function bn(a,b,c){var d;if((d=b.Ac)!=null&&d.Pe)try{var e,f=(e=a.cache)==null?void 0:Dm(e,en(a,b),b.Ac.Od);return f?c?yl(a.logger,function(){return Fd(f,2)},"a"):f:void 0}catch(g){Ic(a,new I(23,"RXO:read",g))}}
function cn(a,b,c){var d={stack:[],error:void 0,ob:!1};try{if(!b.Rf)throw new I(29,"SDF:notready");return Km(fb(d,new Rm(a.logger,0,a.state)),function(){return en(a,b)},c)}catch(e){d.error=e,d.ob=!0}finally{gb(d)}}
function dn(a,b,c){var d={stack:[],error:void 0,ob:!1};try{var e=$m(a,c);return Km(fb(d,new Om(a.logger,e)),function(){return[]},b)}catch(f){d.error=f,d.ob=!0}finally{gb(d)}}
function en(a,b){return b.ld?b.ld:b.Ha?yl(a.logger,function(){return b.ld=Qm(b.Ha)},"c"):[]}
var Um=function(a){return ye(function(b){if(!Fe(b))return!1;for(var c=y(Object.entries(a)),d=c.next();!d.done;d=c.next()){var e=y(d.value);d=e.next().value;e=e.next().value;if(!(d in b)){if(e.gj===!0)continue;return!1}if(!e(b[d]))return!1}return!0})}({vm:function(a){return ye(function(b){return b instanceof a})}($l)},"");var fn=window;function gn(a){var b=hn;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function jn(){var a=[];gn(function(b){a.push(b)});
return a}
;var hn={Kg:"allow-forms",Lg:"allow-modals",Mg:"allow-orientation-lock",Ng:"allow-pointer-lock",Og:"allow-popups",Pg:"allow-popups-to-escape-sandbox",Qg:"allow-presentation",Rg:"allow-same-origin",Sg:"allow-scripts",Tg:"allow-top-navigation",Ug:"allow-top-navigation-by-user-activation"},kn=Fk(function(){return jn()});
function ln(){var a=mn(),b={};Tb(kn(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function mn(){var a=a===void 0?document:a;return a.createElement("iframe")}
;function nn(a){typeof a=="number"&&(a=Math.round(a)+"px");return a}
;var on=(new Date).getTime();function pn(a){this.D=M(a)}
v(pn,N);var qn=Ph(pn);function rn(a){pk.call(this);var b=this;this.B=this.j=0;this.Ea=a!=null?a:{ra:function(e,f){return setTimeout(e,f)},
sa:function(e){clearTimeout(e)}};
var c,d;this.h=(d=(c=window.navigator)==null?void 0:c.onLine)!=null?d:!0;this.o=function(){return B(function(e){return e.yield(sn(b),0)})};
window.addEventListener("offline",this.o);window.addEventListener("online",this.o);this.B||tn(this)}
v(rn,pk);function un(){var a=vn;rn.instance||(rn.instance=new rn(a));return rn.instance}
rn.prototype.dispose=function(){window.removeEventListener("offline",this.o);window.removeEventListener("online",this.o);this.Ea.sa(this.B);delete rn.instance};
rn.prototype.va=function(){return this.h};
function tn(a){a.B=a.Ea.ra(function(){var b;return B(function(c){if(c.h==1)return a.h?((b=window.navigator)==null?0:b.onLine)?c.v(3):c.yield(sn(a),3):c.yield(sn(a),3);tn(a);c.h=0})},3E4)}
function sn(a,b){return a.u?a.u:a.u=new Promise(function(c){var d,e,f,g;return B(function(h){switch(h.h){case 1:return d=window.AbortController?new window.AbortController:void 0,f=(e=d)==null?void 0:e.signal,g=!1,wa(h,2,3),d&&(a.j=a.Ea.ra(function(){d.abort()},b||2E4)),h.yield(fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:za(h);a.u=void 0;a.j&&(a.Ea.sa(a.j),a.j=0);g!==a.h&&(a.h=g,a.h?qk(a,"networkstatus-online"):qk(a,"networkstatus-offline"));c(g);Aa(h,0);break;case 2:ya(h),g=!1,h.v(3)}})})}
;function wn(){this.data=[];this.h=-1}
wn.prototype.set=function(a,b){b=b===void 0?!0:b;0<=a&&a<52&&Number.isInteger(a)&&this.data[a]!==b&&(this.data[a]=b,this.h=-1)};
wn.prototype.get=function(a){return!!this.data[a]};
function xn(a){a.h===-1&&(a.h=a.data.reduce(function(b,c,d){return b+(c?Math.pow(2,d):0)},0));
return a.h}
;function yn(){this.blockSize=-1}
;function zn(){this.blockSize=-1;this.blockSize=64;this.h=[];this.u=[];this.H=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.o=this.i=0;this.reset()}
ab(zn,yn);zn.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.o=this.i=0};
function An(a,b,c){c||(c=0);var d=a.H;if(typeof b==="string")for(var e=0;e<16;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;e<16;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(b=16;b<80;b++)c=d[b-3]^d[b-8]^d[b-14]^d[b-16],d[b]=(c<<1|c>>>31)&4294967295;b=a.h[0];c=a.h[1];e=a.h[2];for(var f=a.h[3],g=a.h[4],h,k,l=0;l<80;l++)l<40?l<20?(h=f^c&(e^f),k=1518500249):(h=c^e^f,k=1859775393):l<60?(h=c&e|f&(c|e),k=2400959708):(h=c^e^f,k=3395469782),
h=(b<<5|b>>>27)+h+g+k+d[l]&4294967295,g=f,f=e,e=(c<<30|c>>>2)&4294967295,c=b,b=h;a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+e&4294967295;a.h[3]=a.h[3]+f&4294967295;a.h[4]=a.h[4]+g&4294967295}
zn.prototype.update=function(a,b){if(a!=null){b===void 0&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.u,f=this.i;d<b;){if(f==0)for(;d<=c;)An(this,a,d),d+=this.blockSize;if(typeof a==="string")for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){An(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){An(this,e);f=0;break}}this.i=f;this.o+=b}};
zn.prototype.digest=function(){var a=[],b=this.o*8;this.i<56?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;c>=56;c--)this.u[c]=b&255,b/=256;An(this,this.u);for(c=b=0;c<5;c++)for(var d=24;d>=0;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function Bn(a){return typeof a.className=="string"?a.className:a.getAttribute&&a.getAttribute("class")||""}
function Cn(a,b){typeof a.className=="string"?a.className=b:a.setAttribute&&a.setAttribute("class",b)}
function Dn(a,b){a.classList?b=a.classList.contains(b):(a=a.classList?a.classList:Bn(a).match(/\S+/g)||[],b=Sb(a,b)>=0);return b}
function En(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):Dn(a,"inverted-hdpi")&&Cn(a,Array.prototype.filter.call(a.classList?a.classList:Bn(a).match(/\S+/g)||[],function(b){return b!="inverted-hdpi"}).join(" "))}
;function Fn(){}
Fn.prototype.next=function(){return Gn};
var Gn={done:!0,value:void 0};Fn.prototype.zb=function(){return this};function Hn(a){if(a instanceof In||a instanceof Jn||a instanceof Kn)return a;if(typeof a.next=="function")return new In(function(){return a});
if(typeof a[Symbol.iterator]=="function")return new In(function(){return a[Symbol.iterator]()});
if(typeof a.zb=="function")return new In(function(){return a.zb()});
throw Error("Not an iterator or iterable.");}
function In(a){this.h=a}
In.prototype.zb=function(){return new Jn(this.h())};
In.prototype[Symbol.iterator]=function(){return new Kn(this.h())};
In.prototype.i=function(){return new Kn(this.h())};
function Jn(a){this.h=a}
v(Jn,Fn);Jn.prototype.next=function(){return this.h.next()};
Jn.prototype[Symbol.iterator]=function(){return new Kn(this.h)};
Jn.prototype.i=function(){return new Kn(this.h)};
function Kn(a){In.call(this,function(){return a});
this.j=a}
v(Kn,In);Kn.prototype.next=function(){return this.j.next()};function O(a){J.call(this);this.u=1;this.j=[];this.o=0;this.h=[];this.i={};this.B=!!a}
ab(O,J);p=O.prototype;p.subscribe=function(a,b,c){var d=this.i[a];d||(d=this.i[a]=[]);var e=this.u;this.h[e]=a;this.h[e+1]=b;this.h[e+2]=c;this.u=e+3;d.push(e);return e};
p.unsubscribe=function(a,b,c){if(a=this.i[a]){var d=this.h;if(a=a.find(function(e){return d[e+1]==b&&d[e+2]==c}))return this.xc(a)}return!1};
p.xc=function(a){var b=this.h[a];if(b){var c=this.i[b];this.o!=0?(this.j.push(a),this.h[a+1]=function(){}):(c&&Yb(c,a),delete this.h[a],delete this.h[a+1],delete this.h[a+2])}return!!b};
p.yb=function(a,b){var c=this.i[a];if(c){var d=Array(arguments.length-1),e=arguments.length,f;for(f=1;f<e;f++)d[f-1]=arguments[f];if(this.B)for(f=0;f<c.length;f++)e=c[f],Ln(this.h[e+1],this.h[e+2],d);else{this.o++;try{for(f=0,e=c.length;f<e&&!this.I;f++){var g=c[f];this.h[g+1].apply(this.h[g+2],d)}}finally{if(this.o--,this.j.length>0&&this.o==0)for(;c=this.j.pop();)this.xc(c)}}return f!=0}return!1};
function Ln(a,b,c){Bk(function(){a.apply(b,c)})}
p.clear=function(a){if(a){var b=this.i[a];b&&(b.forEach(this.xc,this),delete this.i[a])}else this.h.length=0,this.i={}};
p.X=function(){O.Ca.X.call(this);this.clear();this.j.length=0};function Mn(a){this.h=a}
Mn.prototype.set=function(a,b){b===void 0?this.h.remove(a):this.h.set(a,(new fl).serialize(b))};
Mn.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(b!==null)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
Mn.prototype.remove=function(a){this.h.remove(a)};function Nn(a){this.h=a}
ab(Nn,Mn);function On(a){this.data=a}
function Pn(a){return a===void 0||a instanceof On?a:new On(a)}
Nn.prototype.set=function(a,b){Nn.Ca.set.call(this,a,Pn(b))};
Nn.prototype.i=function(a){a=Nn.Ca.get.call(this,a);if(a===void 0||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
Nn.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,a===void 0)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function Qn(a){this.h=a}
ab(Qn,Nn);Qn.prototype.set=function(a,b,c){if(b=Pn(b)){if(c){if(c<Za()){Qn.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Za()}Qn.Ca.set.call(this,a,b)};
Qn.prototype.i=function(a){var b=Qn.Ca.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Za()||c&&c>Za())Qn.prototype.remove.call(this,a);else return b}};function Rn(){}
;function Sn(){}
ab(Sn,Rn);Sn.prototype[Symbol.iterator]=function(){return Hn(this.zb(!0)).i()};
Sn.prototype.clear=function(){var a=Array.from(this);a=y(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function Tn(a){this.h=a;this.i=null}
ab(Tn,Sn);p=Tn.prototype;p.isAvailable=function(){if(this.i===null){var a=this.h;if(a)try{a.setItem("__sak","1");a.removeItem("__sak");var b=!0}catch(c){b=c instanceof DOMException&&(c.name==="QuotaExceededError"||c.code===22||c.code===1014||c.name==="NS_ERROR_DOM_QUOTA_REACHED")&&a&&a.length!==0}else b=!1;this.i=b}return this.i};
p.set=function(a,b){Un(this);try{this.h.setItem(a,b)}catch(c){if(this.h.length==0)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
p.get=function(a){Un(this);a=this.h.getItem(a);if(typeof a!=="string"&&a!==null)throw"Storage mechanism: Invalid value was encountered";return a};
p.remove=function(a){Un(this);this.h.removeItem(a)};
p.zb=function(a){Un(this);var b=0,c=this.h,d=new Fn;d.next=function(){if(b>=c.length)return Gn;var e=c.key(b++);if(a)return{value:e,done:!1};e=c.getItem(e);if(typeof e!=="string")throw"Storage mechanism: Invalid value was encountered";return{value:e,done:!1}};
return d};
p.clear=function(){Un(this);this.h.clear()};
p.key=function(a){Un(this);return this.h.key(a)};
function Un(a){if(a.h==null)throw Error("Storage mechanism: Storage unavailable");a.isAvailable()||ed(Error("Storage mechanism: Storage unavailable"))}
;function Vn(){var a=null;try{a=D.localStorage||null}catch(b){}Tn.call(this,a)}
ab(Vn,Tn);function Wn(a,b){this.i=a;this.h=b+"::"}
ab(Wn,Sn);Wn.prototype.set=function(a,b){this.i.set(this.h+a,b)};
Wn.prototype.get=function(a){return this.i.get(this.h+a)};
Wn.prototype.remove=function(a){this.i.remove(this.h+a)};
Wn.prototype.zb=function(a){var b=this.i[Symbol.iterator](),c=this,d=new Fn;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.h.length)!=c.h;){e=b.next();if(e.done)return e;e=e.value}return{value:a?e.slice(c.h.length):c.i.get(e),done:!1}};
return d};function Xn(a){if(a.bb&&typeof a.bb=="function")return a.bb();if(typeof Map!=="undefined"&&a instanceof Map||typeof Set!=="undefined"&&a instanceof Set)return Array.from(a.values());if(typeof a==="string")return a.split("");if(Oa(a)){for(var b=[],c=a.length,d=0;d<c;d++)b.push(a[d]);return b}return Fi(a)}
function Yn(a){if(a.ec&&typeof a.ec=="function")return a.ec();if(!a.bb||typeof a.bb!="function"){if(typeof Map!=="undefined"&&a instanceof Map)return Array.from(a.keys());if(!(typeof Set!=="undefined"&&a instanceof Set)){if(Oa(a)||typeof a==="string"){var b=[];a=a.length;for(var c=0;c<a;c++)b.push(c);return b}b=[];c=0;for(var d in a)b[c++]=d;return b}}}
function Zn(a,b,c){if(a.forEach&&typeof a.forEach=="function")a.forEach(b,c);else if(Oa(a)||typeof a==="string")Array.prototype.forEach.call(a,b,c);else for(var d=Yn(a),e=Xn(a),f=e.length,g=0;g<f;g++)b.call(c,e[g],d&&d[g],a)}
;function $n(a){this.i=this.B=this.j="";this.G=null;this.u=this.h="";this.o=!1;var b;a instanceof $n?(this.o=a.o,ao(this,a.j),this.B=a.B,this.i=a.i,bo(this,a.G),this.h=a.h,co(this,a.H.clone()),this.u=a.u):a&&(b=String(a).match(jc))?(this.o=!1,ao(this,b[1]||"",!0),this.B=eo(b[2]||""),this.i=eo(b[3]||"",!0),bo(this,b[4]),this.h=eo(b[5]||"",!0),co(this,b[6]||"",!0),this.u=eo(b[7]||"")):(this.o=!1,this.H=new fo(null,this.o))}
$n.prototype.toString=function(){var a=[],b=this.j;b&&a.push(go(b,ho,!0),":");var c=this.i;if(c||b=="file")a.push("//"),(b=this.B)&&a.push(go(b,ho,!0),"@"),a.push(encodeURIComponent(String(c)).replace(/%25([0-9a-fA-F]{2})/g,"%$1")),c=this.G,c!=null&&a.push(":",String(c));if(c=this.h)this.i&&c.charAt(0)!="/"&&a.push("/"),a.push(go(c,c.charAt(0)=="/"?io:jo,!0));(c=this.H.toString())&&a.push("?",c);(c=this.u)&&a.push("#",go(c,ko));return a.join("")};
$n.prototype.resolve=function(a){var b=this.clone(),c=!!a.j;c?ao(b,a.j):c=!!a.B;c?b.B=a.B:c=!!a.i;c?b.i=a.i:c=a.G!=null;var d=a.h;if(c)bo(b,a.G);else if(c=!!a.h){if(d.charAt(0)!="/")if(this.i&&!this.h)d="/"+d;else{var e=b.h.lastIndexOf("/");e!=-1&&(d=b.h.slice(0,e+1)+d)}e=d;if(e==".."||e==".")d="";else if(e.indexOf("./")!=-1||e.indexOf("/.")!=-1){d=e.lastIndexOf("/",0)==0;e=e.split("/");for(var f=[],g=0;g<e.length;){var h=e[g++];h=="."?d&&g==e.length&&f.push(""):h==".."?((f.length>1||f.length==1&&
f[0]!="")&&f.pop(),d&&g==e.length&&f.push("")):(f.push(h),d=!0)}d=f.join("/")}else d=e}c?b.h=d:c=a.H.toString()!=="";c?co(b,a.H.clone()):c=!!a.u;c&&(b.u=a.u);return b};
$n.prototype.clone=function(){return new $n(this)};
function ao(a,b,c){a.j=c?eo(b,!0):b;a.j&&(a.j=a.j.replace(/:$/,""))}
function bo(a,b){if(b){b=Number(b);if(isNaN(b)||b<0)throw Error("Bad port number "+b);a.G=b}else a.G=null}
function co(a,b,c){b instanceof fo?(a.H=b,lo(a.H,a.o)):(c||(b=go(b,mo)),a.H=new fo(b,a.o))}
function eo(a,b){return a?b?decodeURI(a.replace(/%25/g,"%2525")):decodeURIComponent(a):""}
function go(a,b,c){return typeof a==="string"?(a=encodeURI(a).replace(b,no),c&&(a=a.replace(/%25([0-9a-fA-F]{2})/g,"%$1")),a):null}
function no(a){a=a.charCodeAt(0);return"%"+(a>>4&15).toString(16)+(a&15).toString(16)}
var ho=/[#\/\?@]/g,jo=/[#\?:]/g,io=/[#\?]/g,mo=/[#\?@]/g,ko=/#/g;function fo(a,b){this.i=this.h=null;this.j=a||null;this.o=!!b}
function oo(a){a.h||(a.h=new Map,a.i=0,a.j&&pc(a.j,function(b,c){a.add(hc(b),c)}))}
p=fo.prototype;p.add=function(a,b){oo(this);this.j=null;a=po(this,a);var c=this.h.get(a);c||this.h.set(a,c=[]);c.push(b);this.i=this.i+1;return this};
p.remove=function(a){oo(this);a=po(this,a);return this.h.has(a)?(this.j=null,this.i=this.i-this.h.get(a).length,this.h.delete(a)):!1};
p.clear=function(){this.h=this.j=null;this.i=0};
function qo(a,b){oo(a);b=po(a,b);return a.h.has(b)}
p.forEach=function(a,b){oo(this);this.h.forEach(function(c,d){c.forEach(function(e){a.call(b,e,d,this)},this)},this)};
p.ec=function(){oo(this);for(var a=Array.from(this.h.values()),b=Array.from(this.h.keys()),c=[],d=0;d<b.length;d++)for(var e=a[d],f=0;f<e.length;f++)c.push(b[d]);return c};
p.bb=function(a){oo(this);var b=[];if(typeof a==="string")qo(this,a)&&(b=b.concat(this.h.get(po(this,a))));else{a=Array.from(this.h.values());for(var c=0;c<a.length;c++)b=b.concat(a[c])}return b};
p.set=function(a,b){oo(this);this.j=null;a=po(this,a);qo(this,a)&&(this.i=this.i-this.h.get(a).length);this.h.set(a,[b]);this.i=this.i+1;return this};
p.get=function(a,b){if(!a)return b;a=this.bb(a);return a.length>0?String(a[0]):b};
p.toString=function(){if(this.j)return this.j;if(!this.h)return"";for(var a=[],b=Array.from(this.h.keys()),c=0;c<b.length;c++){var d=b[c],e=encodeURIComponent(String(d));d=this.bb(d);for(var f=0;f<d.length;f++){var g=e;d[f]!==""&&(g+="="+encodeURIComponent(String(d[f])));a.push(g)}}return this.j=a.join("&")};
p.clone=function(){var a=new fo;a.j=this.j;this.h&&(a.h=new Map(this.h),a.i=this.i);return a};
function po(a,b){b=String(b);a.o&&(b=b.toLowerCase());return b}
function lo(a,b){b&&!a.o&&(oo(a),a.j=null,a.h.forEach(function(c,d){var e=d.toLowerCase();d!=e&&(this.remove(d),this.remove(e),c.length>0&&(this.j=null,this.h.set(po(this,e),Zb(c)),this.i=this.i+c.length))},a));
a.o=b}
p.extend=function(a){for(var b=0;b<arguments.length;b++)Zn(arguments[b],function(c,d){this.add(d,c)},this)};/*

 (The MIT License)

 Copyright (C) 2014 by Vitaly Puzrin

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 -----------------------------------------------------------------------------
 Ported from zlib, which is under the following license
 https://github.com/madler/zlib/blob/master/zlib.h

 zlib.h -- interface of the 'zlib' general purpose compression library
   version 1.2.8, April 28th, 2013
   Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu
   The data format used by the zlib library is described by RFCs (Request for
   Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
   (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/
var P={},ro=typeof Uint8Array!=="undefined"&&typeof Uint16Array!=="undefined"&&typeof Int32Array!=="undefined";P.assign=function(a){for(var b=Array.prototype.slice.call(arguments,1);b.length;){var c=b.shift();if(c){if(typeof c!=="object")throw new TypeError(c+"must be non-object");for(var d in c)Object.prototype.hasOwnProperty.call(c,d)&&(a[d]=c[d])}}return a};
P.Dd=function(a,b){if(a.length===b)return a;if(a.subarray)return a.subarray(0,b);a.length=b;return a};
var so={Ab:function(a,b,c,d,e){if(b.subarray&&a.subarray)a.set(b.subarray(c,c+d),e);else for(var f=0;f<d;f++)a[e+f]=b[c+f]},
Sd:function(a){var b,c;var d=c=0;for(b=a.length;d<b;d++)c+=a[d].length;var e=new Uint8Array(c);d=c=0;for(b=a.length;d<b;d++){var f=a[d];e.set(f,c);c+=f.length}return e}},to={Ab:function(a,b,c,d,e){for(var f=0;f<d;f++)a[e+f]=b[c+f]},
Sd:function(a){return[].concat.apply([],a)}};
P.ug=function(){ro?(P.xb=Uint8Array,P.Ma=Uint16Array,P.Ie=Int32Array,P.assign(P,so)):(P.xb=Array,P.Ma=Array,P.Ie=Array,P.assign(P,to))};
P.ug();var uo=!0;try{new Uint8Array(1)}catch(a){uo=!1}
function vo(a){var b,c,d=a.length,e=0;for(b=0;b<d;b++){var f=a.charCodeAt(b);if((f&64512)===55296&&b+1<d){var g=a.charCodeAt(b+1);(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)}e+=f<128?1:f<2048?2:f<65536?3:4}var h=new P.xb(e);for(b=c=0;c<e;b++)f=a.charCodeAt(b),(f&64512)===55296&&b+1<d&&(g=a.charCodeAt(b+1),(g&64512)===56320&&(f=65536+(f-55296<<10)+(g-56320),b++)),f<128?h[c++]=f:(f<2048?h[c++]=192|f>>>6:(f<65536?h[c++]=224|f>>>12:(h[c++]=240|f>>>18,h[c++]=128|f>>>12&63),h[c++]=128|f>>>
6&63),h[c++]=128|f&63);return h}
;var wo={};wo=function(a,b,c,d){var e=a&65535|0;a=a>>>16&65535|0;for(var f;c!==0;){f=c>2E3?2E3:c;c-=f;do e=e+b[d++]|0,a=a+e|0;while(--f);e%=65521;a%=65521}return e|a<<16|0};for(var xo={},yo,zo=[],Ao=0;Ao<256;Ao++){yo=Ao;for(var Bo=0;Bo<8;Bo++)yo=yo&1?3988292384^yo>>>1:yo>>>1;zo[Ao]=yo}xo=function(a,b,c,d){c=d+c;for(a^=-1;d<c;d++)a=a>>>8^zo[(a^b[d])&255];return a^-1};var Co={};Co={2:"need dictionary",1:"stream end",0:"","-1":"file error","-2":"stream error","-3":"data error","-4":"insufficient memory","-5":"buffer error","-6":"incompatible version"};function Do(a){for(var b=a.length;--b>=0;)a[b]=0}
var Eo=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0],Fo=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],Go=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7],Ho=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15],Io=Array(576);Do(Io);var Jo=Array(60);Do(Jo);var Ko=Array(512);Do(Ko);var Lo=Array(256);Do(Lo);var Mo=Array(29);Do(Mo);var No=Array(30);Do(No);function Oo(a,b,c,d,e){this.te=a;this.qf=b;this.pf=c;this.bf=d;this.Of=e;this.Vd=a&&a.length}
var Po,Qo,Ro;function So(a,b){this.Rd=a;this.Kb=0;this.jb=b}
function To(a,b){a.da[a.pending++]=b&255;a.da[a.pending++]=b>>>8&255}
function Uo(a,b,c){a.ia>16-c?(a.pa|=b<<a.ia&65535,To(a,a.pa),a.pa=b>>16-a.ia,a.ia+=c-16):(a.pa|=b<<a.ia&65535,a.ia+=c)}
function Vo(a,b,c){Uo(a,c[b*2],c[b*2+1])}
function Wo(a,b){var c=0;do c|=a&1,a>>>=1,c<<=1;while(--b>0);return c>>>1}
function Xo(a,b,c){var d=Array(16),e=0,f;for(f=1;f<=15;f++)d[f]=e=e+c[f-1]<<1;for(c=0;c<=b;c++)e=a[c*2+1],e!==0&&(a[c*2]=Wo(d[e]++,e))}
function Yo(a){var b;for(b=0;b<286;b++)a.ta[b*2]=0;for(b=0;b<30;b++)a.nb[b*2]=0;for(b=0;b<19;b++)a.ja[b*2]=0;a.ta[512]=1;a.Ta=a.Qb=0;a.Aa=a.matches=0}
function Zo(a){a.ia>8?To(a,a.pa):a.ia>0&&(a.da[a.pending++]=a.pa);a.pa=0;a.ia=0}
function $o(a,b,c){Zo(a);To(a,c);To(a,~c);P.Ab(a.da,a.window,b,c,a.pending);a.pending+=c}
function ap(a,b,c,d){var e=b*2,f=c*2;return a[e]<a[f]||a[e]===a[f]&&d[b]<=d[c]}
function bp(a,b,c){for(var d=a.ea[c],e=c<<1;e<=a.Ra;){e<a.Ra&&ap(b,a.ea[e+1],a.ea[e],a.depth)&&e++;if(ap(b,d,a.ea[e],a.depth))break;a.ea[c]=a.ea[e];c=e;e<<=1}a.ea[c]=d}
function cp(a,b,c){var d=0;if(a.Aa!==0){do{var e=a.da[a.ac+d*2]<<8|a.da[a.ac+d*2+1];var f=a.da[a.rd+d];d++;if(e===0)Vo(a,f,b);else{var g=Lo[f];Vo(a,g+256+1,b);var h=Eo[g];h!==0&&(f-=Mo[g],Uo(a,f,h));e--;g=e<256?Ko[e]:Ko[256+(e>>>7)];Vo(a,g,c);h=Fo[g];h!==0&&(e-=No[g],Uo(a,e,h))}}while(d<a.Aa)}Vo(a,256,b)}
function dp(a,b){var c=b.Rd,d=b.jb.te,e=b.jb.Vd,f=b.jb.bf,g,h=-1;a.Ra=0;a.Gb=573;for(g=0;g<f;g++)c[g*2]!==0?(a.ea[++a.Ra]=h=g,a.depth[g]=0):c[g*2+1]=0;for(;a.Ra<2;){var k=a.ea[++a.Ra]=h<2?++h:0;c[k*2]=1;a.depth[k]=0;a.Ta--;e&&(a.Qb-=d[k*2+1])}b.Kb=h;for(g=a.Ra>>1;g>=1;g--)bp(a,c,g);k=f;do g=a.ea[1],a.ea[1]=a.ea[a.Ra--],bp(a,c,1),d=a.ea[1],a.ea[--a.Gb]=g,a.ea[--a.Gb]=d,c[k*2]=c[g*2]+c[d*2],a.depth[k]=(a.depth[g]>=a.depth[d]?a.depth[g]:a.depth[d])+1,c[g*2+1]=c[d*2+1]=k,a.ea[1]=k++,bp(a,c,1);while(a.Ra>=
2);a.ea[--a.Gb]=a.ea[1];g=b.Rd;k=b.Kb;d=b.jb.te;e=b.jb.Vd;f=b.jb.qf;var l=b.jb.pf,m=b.jb.Of,n,r=0;for(n=0;n<=15;n++)a.Na[n]=0;g[a.ea[a.Gb]*2+1]=0;for(b=a.Gb+1;b<573;b++){var t=a.ea[b];n=g[g[t*2+1]*2+1]+1;n>m&&(n=m,r++);g[t*2+1]=n;if(!(t>k)){a.Na[n]++;var w=0;t>=l&&(w=f[t-l]);var z=g[t*2];a.Ta+=z*(n+w);e&&(a.Qb+=z*(d[t*2+1]+w))}}if(r!==0){do{for(n=m-1;a.Na[n]===0;)n--;a.Na[n]--;a.Na[n+1]+=2;a.Na[m]--;r-=2}while(r>0);for(n=m;n!==0;n--)for(t=a.Na[n];t!==0;)d=a.ea[--b],d>k||(g[d*2+1]!==n&&(a.Ta+=(n-g[d*
2+1])*g[d*2],g[d*2+1]=n),t--)}Xo(c,h,a.Na)}
function ep(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);b[(c+1)*2+1]=65535;for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];++g<h&&l===f||(g<k?a.ja[l*2]+=g:l!==0?(l!==e&&a.ja[l*2]++,a.ja[32]++):g<=10?a.ja[34]++:a.ja[36]++,g=0,e=l,f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4))}}
function fp(a,b,c){var d,e=-1,f=b[1],g=0,h=7,k=4;f===0&&(h=138,k=3);for(d=0;d<=c;d++){var l=f;f=b[(d+1)*2+1];if(!(++g<h&&l===f)){if(g<k){do Vo(a,l,a.ja);while(--g!==0)}else l!==0?(l!==e&&(Vo(a,l,a.ja),g--),Vo(a,16,a.ja),Uo(a,g-3,2)):g<=10?(Vo(a,17,a.ja),Uo(a,g-3,3)):(Vo(a,18,a.ja),Uo(a,g-11,7));g=0;e=l;f===0?(h=138,k=3):l===f?(h=6,k=3):(h=7,k=4)}}}
function gp(a){var b=4093624447,c;for(c=0;c<=31;c++,b>>>=1)if(b&1&&a.ta[c*2]!==0)return 0;if(a.ta[18]!==0||a.ta[20]!==0||a.ta[26]!==0)return 1;for(c=32;c<256;c++)if(a.ta[c*2]!==0)return 1;return 0}
var hp=!1;function ip(a,b,c){a.da[a.ac+a.Aa*2]=b>>>8&255;a.da[a.ac+a.Aa*2+1]=b&255;a.da[a.rd+a.Aa]=c&255;a.Aa++;b===0?a.ta[c*2]++:(a.matches++,b--,a.ta[(Lo[c]+256+1)*2]++,a.nb[(b<256?Ko[b]:Ko[256+(b>>>7)])*2]++);return a.Aa===a.hc-1}
;function jp(a,b){a.msg=Co[b];return b}
function kp(a){for(var b=a.length;--b>=0;)a[b]=0}
function lp(a){var b=a.state,c=b.pending;c>a.T&&(c=a.T);c!==0&&(P.Ab(a.output,b.da,b.nc,c,a.Mb),a.Mb+=c,b.nc+=c,a.Fd+=c,a.T-=c,b.pending-=c,b.pending===0&&(b.nc=0))}
function mp(a,b){var c=a.xa>=0?a.xa:-1,d=a.A-a.xa,e=0;if(a.level>0){a.R.jd===2&&(a.R.jd=gp(a));dp(a,a.Kc);dp(a,a.Fc);ep(a,a.ta,a.Kc.Kb);ep(a,a.nb,a.Fc.Kb);dp(a,a.Nd);for(e=18;e>=3&&a.ja[Ho[e]*2+1]===0;e--);a.Ta+=3*(e+1)+5+5+4;var f=a.Ta+3+7>>>3;var g=a.Qb+3+7>>>3;g<=f&&(f=g)}else f=g=d+5;if(d+4<=f&&c!==-1)Uo(a,b?1:0,3),$o(a,c,d);else if(a.strategy===4||g===f)Uo(a,2+(b?1:0),3),cp(a,Io,Jo);else{Uo(a,4+(b?1:0),3);c=a.Kc.Kb+1;d=a.Fc.Kb+1;e+=1;Uo(a,c-257,5);Uo(a,d-1,5);Uo(a,e-4,4);for(f=0;f<e;f++)Uo(a,
a.ja[Ho[f]*2+1],3);fp(a,a.ta,c-1);fp(a,a.nb,d-1);cp(a,a.ta,a.nb)}Yo(a);b&&Zo(a);a.xa=a.A;lp(a.R)}
function R(a,b){a.da[a.pending++]=b}
function np(a,b){a.da[a.pending++]=b>>>8&255;a.da[a.pending++]=b&255}
function op(a,b){var c=a.Zd,d=a.A,e=a.ya,f=a.ce,g=a.A>a.ma-262?a.A-(a.ma-262):0,h=a.window,k=a.kb,l=a.La,m=a.A+258,n=h[d+e-1],r=h[d+e];a.ya>=a.Ud&&(c>>=2);f>a.F&&(f=a.F);do{var t=b;if(h[t+e]===r&&h[t+e-1]===n&&h[t]===h[d]&&h[++t]===h[d+1]){d+=2;for(t++;h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&d<m;);t=258-(m-d);d=m-258;if(t>e){a.Jb=b;e=t;if(t>=f)break;n=h[d+e-1];r=h[d+e]}}}while((b=l[b&k])>g&&--c!==0);return e<=
a.F?e:a.F}
function pp(a){var b=a.ma,c;do{var d=a.Ge-a.F-a.A;if(a.A>=b+(b-262)){P.Ab(a.window,a.window,b,b,0);a.Jb-=b;a.A-=b;a.xa-=b;var e=c=a.Jc;do{var f=a.head[--e];a.head[e]=f>=b?f-b:0}while(--c);e=c=b;do f=a.La[--e],a.La[e]=f>=b?f-b:0;while(--c);d+=b}if(a.R.oa===0)break;e=a.R;c=a.window;f=a.A+a.F;var g=e.oa;g>d&&(g=d);g===0?c=0:(e.oa-=g,P.Ab(c,e.input,e.tb,g,f),e.state.wrap===1?e.M=wo(e.M,c,g,f):e.state.wrap===2&&(e.M=xo(e.M,c,g,f)),e.tb+=g,e.wb+=g,c=g);a.F+=c;if(a.F+a.la>=3)for(d=a.A-a.la,a.S=a.window[d],
a.S=(a.S<<a.Qa^a.window[d+1])&a.Pa;a.la&&!(a.S=(a.S<<a.Qa^a.window[d+3-1])&a.Pa,a.La[d&a.kb]=a.head[a.S],a.head[a.S]=d,d++,a.la--,a.F+a.la<3););}while(a.F<262&&a.R.oa!==0)}
function qp(a,b){for(var c;;){if(a.F<262){pp(a);if(a.F<262&&b===0)return 1;if(a.F===0)break}c=0;a.F>=3&&(a.S=(a.S<<a.Qa^a.window[a.A+3-1])&a.Pa,c=a.La[a.A&a.kb]=a.head[a.S],a.head[a.S]=a.A);c!==0&&a.A-c<=a.ma-262&&(a.U=op(a,c));if(a.U>=3)if(c=ip(a,a.A-a.Jb,a.U-3),a.F-=a.U,a.U<=a.td&&a.F>=3){a.U--;do a.A++,a.S=(a.S<<a.Qa^a.window[a.A+3-1])&a.Pa,a.La[a.A&a.kb]=a.head[a.S],a.head[a.S]=a.A;while(--a.U!==0);a.A++}else a.A+=a.U,a.U=0,a.S=a.window[a.A],a.S=(a.S<<a.Qa^a.window[a.A+1])&a.Pa;else c=ip(a,0,
a.window[a.A]),a.F--,a.A++;if(c&&(mp(a,!1),a.R.T===0))return 1}a.la=a.A<2?a.A:2;return b===4?(mp(a,!0),a.R.T===0?3:4):a.Aa&&(mp(a,!1),a.R.T===0)?1:2}
function rp(a,b){for(var c,d;;){if(a.F<262){pp(a);if(a.F<262&&b===0)return 1;if(a.F===0)break}c=0;a.F>=3&&(a.S=(a.S<<a.Qa^a.window[a.A+3-1])&a.Pa,c=a.La[a.A&a.kb]=a.head[a.S],a.head[a.S]=a.A);a.ya=a.U;a.ge=a.Jb;a.U=2;c!==0&&a.ya<a.td&&a.A-c<=a.ma-262&&(a.U=op(a,c),a.U<=5&&(a.strategy===1||a.U===3&&a.A-a.Jb>4096)&&(a.U=2));if(a.ya>=3&&a.U<=a.ya){d=a.A+a.F-3;c=ip(a,a.A-1-a.ge,a.ya-3);a.F-=a.ya-1;a.ya-=2;do++a.A<=d&&(a.S=(a.S<<a.Qa^a.window[a.A+3-1])&a.Pa,a.La[a.A&a.kb]=a.head[a.S],a.head[a.S]=a.A);
while(--a.ya!==0);a.rb=0;a.U=2;a.A++;if(c&&(mp(a,!1),a.R.T===0))return 1}else if(a.rb){if((c=ip(a,0,a.window[a.A-1]))&&mp(a,!1),a.A++,a.F--,a.R.T===0)return 1}else a.rb=1,a.A++,a.F--}a.rb&&(ip(a,0,a.window[a.A-1]),a.rb=0);a.la=a.A<2?a.A:2;return b===4?(mp(a,!0),a.R.T===0?3:4):a.Aa&&(mp(a,!1),a.R.T===0)?1:2}
function sp(a,b){for(var c,d,e,f=a.window;;){if(a.F<=258){pp(a);if(a.F<=258&&b===0)return 1;if(a.F===0)break}a.U=0;if(a.F>=3&&a.A>0&&(d=a.A-1,c=f[d],c===f[++d]&&c===f[++d]&&c===f[++d])){for(e=a.A+258;c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&d<e;);a.U=258-(e-d);a.U>a.F&&(a.U=a.F)}a.U>=3?(c=ip(a,1,a.U-3),a.F-=a.U,a.A+=a.U,a.U=0):(c=ip(a,0,a.window[a.A]),a.F--,a.A++);if(c&&(mp(a,!1),a.R.T===0))return 1}a.la=0;return b===4?(mp(a,!0),a.R.T===0?3:4):
a.Aa&&(mp(a,!1),a.R.T===0)?1:2}
function tp(a,b){for(var c;;){if(a.F===0&&(pp(a),a.F===0)){if(b===0)return 1;break}a.U=0;c=ip(a,0,a.window[a.A]);a.F--;a.A++;if(c&&(mp(a,!1),a.R.T===0))return 1}a.la=0;return b===4?(mp(a,!0),a.R.T===0?3:4):a.Aa&&(mp(a,!1),a.R.T===0)?1:2}
function up(a,b,c,d,e){this.yf=a;this.Nf=b;this.Tf=c;this.Mf=d;this.uf=e}
var vp;vp=[new up(0,0,0,0,function(a,b){var c=65535;for(c>a.Ba-5&&(c=a.Ba-5);;){if(a.F<=1){pp(a);if(a.F===0&&b===0)return 1;if(a.F===0)break}a.A+=a.F;a.F=0;var d=a.xa+c;if(a.A===0||a.A>=d)if(a.F=a.A-d,a.A=d,mp(a,!1),a.R.T===0)return 1;if(a.A-a.xa>=a.ma-262&&(mp(a,!1),a.R.T===0))return 1}a.la=0;if(b===4)return mp(a,!0),a.R.T===0?3:4;a.A>a.xa&&mp(a,!1);return 1}),
new up(4,4,8,4,qp),new up(4,5,16,8,qp),new up(4,6,32,32,qp),new up(4,4,16,16,rp),new up(8,16,32,32,rp),new up(8,16,128,128,rp),new up(8,32,128,256,rp),new up(32,128,258,1024,rp),new up(32,258,258,4096,rp)];
function wp(){this.R=null;this.status=0;this.da=null;this.wrap=this.pending=this.nc=this.Ba=0;this.J=null;this.Da=0;this.method=8;this.Ib=-1;this.kb=this.Id=this.ma=0;this.window=null;this.Ge=0;this.head=this.La=null;this.ce=this.Ud=this.strategy=this.level=this.td=this.Zd=this.ya=this.F=this.Jb=this.A=this.rb=this.ge=this.U=this.xa=this.Qa=this.Pa=this.pd=this.Jc=this.S=0;this.ta=new P.Ma(1146);this.nb=new P.Ma(122);this.ja=new P.Ma(78);kp(this.ta);kp(this.nb);kp(this.ja);this.Nd=this.Fc=this.Kc=
null;this.Na=new P.Ma(16);this.ea=new P.Ma(573);kp(this.ea);this.Gb=this.Ra=0;this.depth=new P.Ma(573);kp(this.depth);this.ia=this.pa=this.la=this.matches=this.Qb=this.Ta=this.ac=this.Aa=this.hc=this.rd=0}
function xp(a,b){if(!a||!a.state||b>5||b<0)return a?jp(a,-2):-2;var c=a.state;if(!a.output||!a.input&&a.oa!==0||c.status===666&&b!==4)return jp(a,a.T===0?-5:-2);c.R=a;var d=c.Ib;c.Ib=b;if(c.status===42)if(c.wrap===2)a.M=0,R(c,31),R(c,139),R(c,8),c.J?(R(c,(c.J.text?1:0)+(c.J.cb?2:0)+(c.J.extra?4:0)+(c.J.name?8:0)+(c.J.comment?16:0)),R(c,c.J.time&255),R(c,c.J.time>>8&255),R(c,c.J.time>>16&255),R(c,c.J.time>>24&255),R(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),R(c,c.J.os&255),c.J.extra&&c.J.extra.length&&
(R(c,c.J.extra.length&255),R(c,c.J.extra.length>>8&255)),c.J.cb&&(a.M=xo(a.M,c.da,c.pending,0)),c.Da=0,c.status=69):(R(c,0),R(c,0),R(c,0),R(c,0),R(c,0),R(c,c.level===9?2:c.strategy>=2||c.level<2?4:0),R(c,3),c.status=113);else{var e=8+(c.Id-8<<4)<<8;e|=(c.strategy>=2||c.level<2?0:c.level<6?1:c.level===6?2:3)<<6;c.A!==0&&(e|=32);c.status=113;np(c,e+(31-e%31));c.A!==0&&(np(c,a.M>>>16),np(c,a.M&65535));a.M=1}if(c.status===69)if(c.J.extra){for(e=c.pending;c.Da<(c.J.extra.length&65535)&&(c.pending!==c.Ba||
(c.J.cb&&c.pending>e&&(a.M=xo(a.M,c.da,c.pending-e,e)),lp(a),e=c.pending,c.pending!==c.Ba));)R(c,c.J.extra[c.Da]&255),c.Da++;c.J.cb&&c.pending>e&&(a.M=xo(a.M,c.da,c.pending-e,e));c.Da===c.J.extra.length&&(c.Da=0,c.status=73)}else c.status=73;if(c.status===73)if(c.J.name){e=c.pending;do{if(c.pending===c.Ba&&(c.J.cb&&c.pending>e&&(a.M=xo(a.M,c.da,c.pending-e,e)),lp(a),e=c.pending,c.pending===c.Ba)){var f=1;break}f=c.Da<c.J.name.length?c.J.name.charCodeAt(c.Da++)&255:0;R(c,f)}while(f!==0);c.J.cb&&c.pending>
e&&(a.M=xo(a.M,c.da,c.pending-e,e));f===0&&(c.Da=0,c.status=91)}else c.status=91;if(c.status===91)if(c.J.comment){e=c.pending;do{if(c.pending===c.Ba&&(c.J.cb&&c.pending>e&&(a.M=xo(a.M,c.da,c.pending-e,e)),lp(a),e=c.pending,c.pending===c.Ba)){f=1;break}f=c.Da<c.J.comment.length?c.J.comment.charCodeAt(c.Da++)&255:0;R(c,f)}while(f!==0);c.J.cb&&c.pending>e&&(a.M=xo(a.M,c.da,c.pending-e,e));f===0&&(c.status=103)}else c.status=103;c.status===103&&(c.J.cb?(c.pending+2>c.Ba&&lp(a),c.pending+2<=c.Ba&&(R(c,
a.M&255),R(c,a.M>>8&255),a.M=0,c.status=113)):c.status=113);if(c.pending!==0){if(lp(a),a.T===0)return c.Ib=-1,0}else if(a.oa===0&&(b<<1)-(b>4?9:0)<=(d<<1)-(d>4?9:0)&&b!==4)return jp(a,-5);if(c.status===666&&a.oa!==0)return jp(a,-5);if(a.oa!==0||c.F!==0||b!==0&&c.status!==666){d=c.strategy===2?tp(c,b):c.strategy===3?sp(c,b):vp[c.level].uf(c,b);if(d===3||d===4)c.status=666;if(d===1||d===3)return a.T===0&&(c.Ib=-1),0;if(d===2&&(b===1?(Uo(c,2,3),Vo(c,256,Io),c.ia===16?(To(c,c.pa),c.pa=0,c.ia=0):c.ia>=
8&&(c.da[c.pending++]=c.pa&255,c.pa>>=8,c.ia-=8)):b!==5&&(Uo(c,0,3),$o(c,0,0),b===3&&(kp(c.head),c.F===0&&(c.A=0,c.xa=0,c.la=0))),lp(a),a.T===0))return c.Ib=-1,0}if(b!==4)return 0;if(c.wrap<=0)return 1;c.wrap===2?(R(c,a.M&255),R(c,a.M>>8&255),R(c,a.M>>16&255),R(c,a.M>>24&255),R(c,a.wb&255),R(c,a.wb>>8&255),R(c,a.wb>>16&255),R(c,a.wb>>24&255)):(np(c,a.M>>>16),np(c,a.M&65535));lp(a);c.wrap>0&&(c.wrap=-c.wrap);return c.pending!==0?0:1}
;var yp={};yp=function(){this.input=null;this.wb=this.oa=this.tb=0;this.output=null;this.Fd=this.T=this.Mb=0;this.msg="";this.state=null;this.jd=2;this.M=0};var zp=Object.prototype.toString;
function Ap(a){if(!(this instanceof Ap))return new Ap(a);a=this.options=P.assign({level:-1,method:8,chunkSize:16384,windowBits:15,memLevel:8,strategy:0,to:""},a||{});a.raw&&a.windowBits>0?a.windowBits=-a.windowBits:a.gzip&&a.windowBits>0&&a.windowBits<16&&(a.windowBits+=16);this.err=0;this.msg="";this.ended=!1;this.chunks=[];this.R=new yp;this.R.T=0;var b=this.R;var c=a.level,d=a.method,e=a.windowBits,f=a.memLevel,g=a.strategy;if(b){var h=1;c===-1&&(c=6);e<0?(h=0,e=-e):e>15&&(h=2,e-=16);if(f<1||f>
9||d!==8||e<8||e>15||c<0||c>9||g<0||g>4)b=jp(b,-2);else{e===8&&(e=9);var k=new wp;b.state=k;k.R=b;k.wrap=h;k.J=null;k.Id=e;k.ma=1<<k.Id;k.kb=k.ma-1;k.pd=f+7;k.Jc=1<<k.pd;k.Pa=k.Jc-1;k.Qa=~~((k.pd+3-1)/3);k.window=new P.xb(k.ma*2);k.head=new P.Ma(k.Jc);k.La=new P.Ma(k.ma);k.hc=1<<f+6;k.Ba=k.hc*4;k.da=new P.xb(k.Ba);k.ac=1*k.hc;k.rd=3*k.hc;k.level=c;k.strategy=g;k.method=d;if(b&&b.state){b.wb=b.Fd=0;b.jd=2;c=b.state;c.pending=0;c.nc=0;c.wrap<0&&(c.wrap=-c.wrap);c.status=c.wrap?42:113;b.M=c.wrap===2?
0:1;c.Ib=0;if(!hp){d=Array(16);for(f=g=0;f<28;f++)for(Mo[f]=g,e=0;e<1<<Eo[f];e++)Lo[g++]=f;Lo[g-1]=f;for(f=g=0;f<16;f++)for(No[f]=g,e=0;e<1<<Fo[f];e++)Ko[g++]=f;for(g>>=7;f<30;f++)for(No[f]=g<<7,e=0;e<1<<Fo[f]-7;e++)Ko[256+g++]=f;for(e=0;e<=15;e++)d[e]=0;for(e=0;e<=143;)Io[e*2+1]=8,e++,d[8]++;for(;e<=255;)Io[e*2+1]=9,e++,d[9]++;for(;e<=279;)Io[e*2+1]=7,e++,d[7]++;for(;e<=287;)Io[e*2+1]=8,e++,d[8]++;Xo(Io,287,d);for(e=0;e<30;e++)Jo[e*2+1]=5,Jo[e*2]=Wo(e,5);Po=new Oo(Io,Eo,257,286,15);Qo=new Oo(Jo,
Fo,0,30,15);Ro=new Oo([],Go,0,19,7);hp=!0}c.Kc=new So(c.ta,Po);c.Fc=new So(c.nb,Qo);c.Nd=new So(c.ja,Ro);c.pa=0;c.ia=0;Yo(c);c=0}else c=jp(b,-2);c===0&&(b=b.state,b.Ge=2*b.ma,kp(b.head),b.td=vp[b.level].Nf,b.Ud=vp[b.level].yf,b.ce=vp[b.level].Tf,b.Zd=vp[b.level].Mf,b.A=0,b.xa=0,b.F=0,b.la=0,b.U=b.ya=2,b.rb=0,b.S=0);b=c}}else b=-2;if(b!==0)throw Error(Co[b]);a.header&&(b=this.R)&&b.state&&b.state.wrap===2&&(b.state.J=a.header);if(a.dictionary){var l;typeof a.dictionary==="string"?l=vo(a.dictionary):
zp.call(a.dictionary)==="[object ArrayBuffer]"?l=new Uint8Array(a.dictionary):l=a.dictionary;a=this.R;f=l;g=f.length;if(a&&a.state)if(l=a.state,b=l.wrap,b===2||b===1&&l.status!==42||l.F)b=-2;else{b===1&&(a.M=wo(a.M,f,g,0));l.wrap=0;g>=l.ma&&(b===0&&(kp(l.head),l.A=0,l.xa=0,l.la=0),c=new P.xb(l.ma),P.Ab(c,f,g-l.ma,l.ma,0),f=c,g=l.ma);c=a.oa;d=a.tb;e=a.input;a.oa=g;a.tb=0;a.input=f;for(pp(l);l.F>=3;){f=l.A;g=l.F-2;do l.S=(l.S<<l.Qa^l.window[f+3-1])&l.Pa,l.La[f&l.kb]=l.head[l.S],l.head[l.S]=f,f++;while(--g);
l.A=f;l.F=2;pp(l)}l.A+=l.F;l.xa=l.A;l.la=l.F;l.F=0;l.U=l.ya=2;l.rb=0;a.tb=d;a.input=e;a.oa=c;l.wrap=b;b=0}else b=-2;if(b!==0)throw Error(Co[b]);this.Ri=!0}}
Ap.prototype.push=function(a,b){var c=this.R,d=this.options.chunkSize;if(this.ended)return!1;var e=b===~~b?b:b===!0?4:0;typeof a==="string"?c.input=vo(a):zp.call(a)==="[object ArrayBuffer]"?c.input=new Uint8Array(a):c.input=a;c.tb=0;c.oa=c.input.length;do{c.T===0&&(c.output=new P.xb(d),c.Mb=0,c.T=d);a=xp(c,e);if(a!==1&&a!==0)return Bp(this,a),this.ended=!0,!1;if(c.T===0||c.oa===0&&(e===4||e===2))if(this.options.to==="string"){var f=P.Dd(c.output,c.Mb);b=f;f=f.length;if(f<65537&&(b.subarray&&uo||!b.subarray))b=
String.fromCharCode.apply(null,P.Dd(b,f));else{for(var g="",h=0;h<f;h++)g+=String.fromCharCode(b[h]);b=g}this.chunks.push(b)}else b=P.Dd(c.output,c.Mb),this.chunks.push(b)}while((c.oa>0||c.T===0)&&a!==1);if(e===4)return(c=this.R)&&c.state?(d=c.state.status,d!==42&&d!==69&&d!==73&&d!==91&&d!==103&&d!==113&&d!==666?a=jp(c,-2):(c.state=null,a=d===113?jp(c,-3):0)):a=-2,Bp(this,a),this.ended=!0,a===0;e===2&&(Bp(this,0),c.T=0);return!0};
function Bp(a,b){b===0&&(a.result=a.options.to==="string"?a.chunks.join(""):P.Sd(a.chunks));a.chunks=[];a.err=b;a.msg=a.R.msg}
function Cp(a,b){b=b||{};b.gzip=!0;b=new Ap(b);b.push(a,!0);if(b.err)throw b.msg||Co[b.err];return b.result}
;function Dp(a){return a?(a=a.privateDoNotAccessOrElseSafeScriptWrappedValue)?Ib(a):null:null}
function Ep(a){return a?(a=a.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue)?pb(a):null:null}
;function Fp(a){return pb(a===null?"null":a===void 0?"undefined":a)}
;function Gp(a){this.name=a}
;var Hp=new Gp("rawColdConfigGroup");var Ip=new Gp("rawHotConfigGroup");function Jp(a){this.D=M(a)}
v(Jp,N);function Kp(a){this.D=M(a)}
v(Kp,N);Kp.prototype.setTrackingParams=function(a){return eg(this,1,ue(a,!1))};var Lp=new Gp("continuationCommand");var Mp=new Gp("webCommandMetadata");var Np=new Gp("signalServiceEndpoint");var Op={Zg:"EMBEDDED_PLAYER_MODE_UNKNOWN",Wg:"EMBEDDED_PLAYER_MODE_DEFAULT",Yg:"EMBEDDED_PLAYER_MODE_PFP",Xg:"EMBEDDED_PLAYER_MODE_PFL"};var Pp=new Gp("feedbackEndpoint");var De={si:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNKNOWN",zh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_FOR_TESTING",Xh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_RESUME_TO_HOME_TTL",ii:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_START_TO_SHORTS_ANALYSIS_SLICE",mh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DEVICE_LAYER_SLICE",ri:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_UNIFIED_LAYER_SLICE",wi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_VISITOR_LAYER_SLICE",gi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SHOW_SHEET_COMMAND_HANDLER_BLOCK",
yi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_MIGRATED_COMPONENT",xi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WIZ_NEXT_CHANNEL_NAME_TOOLTIP",bi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATION_LOCK_SUPPORTED",ki:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_THEATER_MODE_ENABLED",Di:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_PIN_SUGGESTION",Ci:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_LONG_PRESS_EDU_TOAST",Bi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_AMBIENT",li:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TIME_WATCHED_PANEL",
di:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SEARCH_FROM_SEARCH_BAR_OVERLAY",Ei:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_SHOW_VOICE_SEARCH_EDU_TOAST",ji:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SUGGESTED_LANGUAGE_SELECTED",Fi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_TRIGGER_SHORTS_PIP",Gh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IN_ZP_VOICE_CRASHY_SET",Th:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_SUPPRESSED",Sh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_FAST_SWIPE_ALLOWED",Vh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_PULL_TO_REFRESH_ATTEMPT",
zi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_BLOCK_KABUKI",Wh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_TALL_SCREEN",Uh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_REEL_NORMAL_SCREEN",eh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_ENABLED",dh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ACCESSIBILITY_MODE_DISABLED",fh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_AUTOPLAY_ENABLED",gh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_CAST_MATCH_OCCURRED",rh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_ELIGIBLE",uh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ENDSCREEN_TRIGGERED",
Rh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_POSTPLAY_TRIGGERED",Qh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_POSTPLAY_LACT_THRESHOLD_EXCEEDED",Ah:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_MATCHED_ON_REMOTE_CONNECTION",Ch:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHABLE_ON_REMOTE_CONNECTION",Bh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_MISATTRIBUTED_ON_REMOTE_CONNECTION",Fh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_TV_IS_SIGNED_IN_ON_REMOTE_CONNECTION",ni:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TV_START_TYPE_COLD_ON_REMOTE_CONNECTION",
oi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TV_START_TYPE_NON_COLD_ON_REMOTE_CONNECTION",Mh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ON_REMOTE_CONNECTION",kh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_VALID",ih:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_INVALID",jh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_UNDEFINED",hh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_COBALT_PERSISTENT_SETTINGS_TEST_DEFINED",Hh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_LACT_THRESHOLD_EXCEEDED",
ci:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROUND_TRIP_HANDLING_ON_REMOTE_CONNECTION",Eh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHED_ON_REMOTE_CONNECTION_BEFORE_APP_RELOAD",Dh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_IDENTITIES_STATE_SWITCHED_ON_REMOTE_CONNECTION_AFTER_APP_RELOAD",sh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_INELIGIBLE",mi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_TVHTML5_MID_ROLL_THRESHOLD_REACHED",xh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_PENDING",
wh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_ACTIVATED",th:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMC3DS_M2_ELIGIBLE",Zh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATE_DEVICE_TO_LANDSCAPE",ai:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ROTATE_DEVICE_TO_PORTRAIT",qh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EMBEDS_FACEOFF_UI_EVENT",yh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_EXP_COBALT_HTTP3_CONFIG_RECEIVED",ph:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_ELIGIBLE_TO_SUPPRESS_TRANSPORT_CONTROLS_BUTTONS",
ti:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_USER_HAS_THEATER_MODE_COOKIE_ENABLED",oh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DOCUMENT_PICTURE_IN_PICTURE_SUPPORTED",fi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SHORTS_NON_DEFAULT_ASPECT_RATIO",Ph:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_PLAYER_IN_SQUEEZEBACK",Ih:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_LIVE_CREATOR_AR_GIFT_RECEIVED",Yh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_RETURNED_TO_VIDEO_AFTER_FAILED_ATTEMPT_TO_BACKGROUND",Ai:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_WOULD_ENTER_AUTO_ZOOM",
Nh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_PASSIVE_IN_CONTROL",Oh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_PASSIVE_IN_TREATMENT",nh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_DISABLE_PLAYER_OPEN_ON_FULLSCREEN",Lh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_MDX_RECONNECT_WITH_RETRY",hi:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_SINGLE_COLUMN_GRID_TRIGGERED",Kh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_MDX_CONNECTION_TIMEOUT",Jh:"GENERIC_CLIENT_EXPERIMENT_EVENT_TYPE_LIVE_GHOST_LOADING_ELIGIBLE"};var Qp=new Gp("shareEndpoint"),Rp=new Gp("shareEntityEndpoint"),Sp=new Gp("shareEntityServiceEndpoint"),Tp=new Gp("webPlayerShareEntityServiceEndpoint");var Up=new Gp("playlistEditEndpoint");var Vp=new Gp("modifyChannelNotificationPreferenceEndpoint");var Wp=new Gp("undoFeedbackEndpoint");var Xp=new Gp("unsubscribeEndpoint");var Yp=new Gp("subscribeEndpoint");function Zp(){var a=$p;G("yt.ads.biscotti.getId_")||E("yt.ads.biscotti.getId_",a)}
function aq(a){E("yt.ads.biscotti.lastId_",a)}
;function bq(a,b){b.length>1?a[b[0]]=b[1]:b.length===1&&Object.assign(a,b[0])}
;var cq=D.window,dq,eq,fq=(cq==null?void 0:(dq=cq.yt)==null?void 0:dq.config_)||(cq==null?void 0:(eq=cq.ytcfg)==null?void 0:eq.data_)||{};E("yt.config_",fq);function gq(){bq(fq,arguments)}
function S(a,b){return a in fq?fq[a]:b}
function hq(a){var b=fq.EXPERIMENT_FLAGS;return b?b[a]:void 0}
;var iq=[];function jq(a){iq.forEach(function(b){return b(a)})}
function kq(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){lq(b)}}:a}
function lq(a){var b=G("yt.logging.errors.log");b?b(a,"ERROR",void 0,void 0,void 0,void 0,void 0):(b=S("ERRORS",[]),b.push([a,"ERROR",void 0,void 0,void 0,void 0,void 0]),gq("ERRORS",b));jq(a)}
function mq(a,b,c,d,e){var f=G("yt.logging.errors.log");f?f(a,"WARNING",b,c,d,void 0,e):(f=S("ERRORS",[]),f.push([a,"WARNING",b,c,d,void 0,e]),gq("ERRORS",f))}
;var nq=/^[\w.]*$/,oq={q:!0,search_query:!0};function pq(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(f.length===1&&f[0]||f.length===2)try{var g=qq(f[0]||""),h=qq(f[1]||"");if(g in c){var k=c[g];Array.isArray(k)?$b(k,h):c[g]=[k,h]}else c[g]=h}catch(r){var l=r,m=f[0],n=String(pq);l.args=[{key:m,value:f[1],query:a,method:rq===n?"unchanged":n}];oq.hasOwnProperty(m)||mq(l)}}return c}
var rq=String(pq);function sq(a){var b=[];Ei(a,function(c,d){var e=encodeURIComponent(String(d));c=Array.isArray(c)?c:[c];Tb(c,function(f){f==""?b.push(e):b.push(e+"="+encodeURIComponent(String(f)))})});
return b.join("&")}
function tq(a){a.charAt(0)==="?"&&(a=a.substring(1));return pq(a,"&")}
function uq(a){return a.indexOf("?")!==-1?(a=(a||"").split("#")[0],a=a.split("?",2),tq(a.length>1?a[1]:a[0])):{}}
function vq(a,b){return wq(a,b||{},!0)}
function wq(a,b,c){var d=a.split("#",2);a=d[0];d=d.length>1?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=tq(e[1]||"");for(var f in b)!c&&e!==null&&f in e||(e[f]=b[f]);return sc(a,e)+d}
function xq(a){if(!b)var b=window.location.href;var c=a.match(jc)[1]||null,d=lc(a);c&&d?(a=a.match(jc),b=b.match(jc),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?lc(b)===d&&(Number(b.match(jc)[4]||null)||null)===(Number(a.match(jc)[4]||null)||null):!0;return a}
function qq(a){return a&&a.match(nq)?a:hc(a)}
;function yq(a){var b=zq;a=a===void 0?G("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=on;e.flash="0";a:{try{var f=b.h.top.location.href}catch(Ra){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();try{var g=fn.history.length}catch(Ra){g=0}e.u_his=g;var h;e.u_h=(h=fn.screen)==null?void 0:h.height;var k;e.u_w=(k=fn.screen)==null?void 0:k.width;var l;e.u_ah=(l=fn.screen)==null?void 0:l.availHeight;var m;e.u_aw=(m=fn.screen)==null?
void 0:m.availWidth;var n;e.u_cd=(n=fn.screen)==null?void 0:n.colorDepth}catch(Ra){}var r;g=b.h;try{var t=g.screenX;var w=g.screenY}catch(Ra){}try{var z=g.outerWidth;var x=g.outerHeight}catch(Ra){}try{var H=g.innerWidth;var F=g.innerHeight}catch(Ra){}try{var V=g.screenLeft;var na=g.screenTop}catch(Ra){}try{H=g.innerWidth,F=g.innerHeight}catch(Ra){}try{var ka=g.screen.availWidth;var Od=g.screen.availTop}catch(Ra){}t=[V,na,t,w,ka,Od,z,x,H,F];try{var Nb=(b.h.top||window).document,hb=Nb.compatMode=="CSS1Compat"?
Nb.documentElement:Nb.body;var Sa=(new Di(hb.clientWidth,hb.clientHeight)).round()}catch(Ra){Sa=new Di(-12245933,-12245933)}Nb=Sa;Sa={};var Ga=Ga===void 0?D:Ga;hb=new wn;"SVGElement"in Ga&&"createElementNS"in Ga.document&&hb.set(0);w=ln();w["allow-top-navigation-by-user-activation"]&&hb.set(1);w["allow-popups-to-escape-sandbox"]&&hb.set(2);Ga.crypto&&Ga.crypto.subtle&&hb.set(3);"TextDecoder"in Ga&&"TextEncoder"in Ga&&hb.set(4);Ga=xn(hb);Sa.bc=Ga;Sa.bih=Nb.height;Sa.biw=Nb.width;Sa.brdim=t.join();
b=b.i;b=b.prerendering?3:(r={visible:1,hidden:2,prerender:3,preview:4,unloaded:5,"":0}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""])!=null?r:0;r=(Sa.vis=b,Sa.wgl=!!fn.WebGLRenderingContext,Sa);c=d.call(c,e,r);c.ca_type="image";a&&(c.bid=a);return c}
var zq=new function(){var a=window.document;this.h=window;this.i=a};
E("yt.ads_.signals_.getAdSignalsString",function(a){return sq(yq(a))});Za();navigator.userAgent.indexOf(" (CrKey ");var Aq="XMLHttpRequest"in D?function(){return new XMLHttpRequest}:null;
function Bq(){if(!Aq)return null;var a=Aq();return"open"in a?a:null}
function Cq(a){switch(Dq(a)){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
function Dq(a){return a&&"status"in a?a.status:-1}
;function Eq(a,b){typeof a==="function"&&(a=kq(a));return window.setTimeout(a,b)}
;var Fq="client_dev_domain client_dev_expflag client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");[].concat(A(Fq),["client_dev_set_cookie"]);var Gq=Symbol("flagNamePruned");function T(a){a=Hq(a);return typeof a==="string"&&a==="false"?!1:!!a}
function Iq(a,b){a=Hq(a);return a===void 0&&b!==void 0?b:Number(a||0)}
function Jq(){var a=Hq("html5_web_po_experiment_ids");return Array.isArray(a)?Vb(a,function(b){return Number(b||0)}):[Number(a||0)]}
function Kq(a){a=Hq(a);return a!==void 0?String(a):""}
function Hq(a){if(a!==Gq)return S("EXPERIMENT_FLAGS",{})[a]}
function Lq(){for(var a=[],b=S("EXPERIMENTS_FORCED_FLAGS",{}),c=y(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=S("EXPERIMENT_FLAGS",{});d=y(Object.keys(c));for(var e=d.next();!e.done;e=d.next())e=e.value,e.startsWith("force_")&&b[e]===void 0&&a.push({key:e,value:String(c[e])});return a}
;var Mq={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-Goog-AuthUser":"SESSION_INDEX","X-Goog-PageId":"DELEGATED_SESSION_ID"},Nq="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(A(Fq)),Oq=!1;function Pq(a,b,c,d,e,f,g,h,k){function l(){(m&&"readyState"in m?m.readyState:0)===4&&b&&kq(b)(m)}
c=c===void 0?"GET":c;d=d===void 0?"":d;h=h===void 0?!1:h;var m=Bq();if(!m)return null;"onloadend"in m?m.addEventListener("loadend",l,!1):m.onreadystatechange=l;T("debug_forward_web_query_parameters")&&(a=Qq(a));m.open(c,a,!0);f&&(m.responseType=f);g&&(m.withCredentials=!0);c=c==="POST"&&(window.FormData===void 0||!(d instanceof FormData));if(e=Rq(a,e))for(var n in e)m.setRequestHeader(n,e[n]),"content-type"===n.toLowerCase()&&(c=!1);c&&m.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
k&&"onprogress"in m&&(m.onprogress=function(){k(m.responseText)});
if(h&&"setAttributionReporting"in XMLHttpRequest.prototype){a={eventSourceEligible:!0,triggerEligible:!1};try{m.setAttributionReporting(a)}catch(r){mq(r)}}m.send(d);return m}
function Rq(a,b){b=b===void 0?{}:b;var c=xq(a),d=S("INNERTUBE_CLIENT_NAME"),e=T("web_ajax_ignore_global_headers_if_set"),f;for(f in Mq){var g=S(Mq[f]),h=f==="X-Goog-AuthUser"||f==="X-Goog-PageId";f!=="X-Goog-Visitor-Id"||g||(g=S("VISITOR_DATA"));var k;if(!(k=!g)){if(!(k=c||(lc(a)?!1:!0))){k=a;var l;if(l=T("add_auth_headers_to_remarketing_google_dot_com_ping")&&f==="Authorization"&&(d==="TVHTML5"||d==="TVHTML5_UNPLUGGED"||d==="TVHTML5_SIMPLY"))l=lc(k),l=l!==null?l.split(".").reverse():null,l=l===null?
!1:l[1]==="google"?!0:l[2]==="google"?l[0]==="au"&&l[1]==="com"?!0:l[0]==="uk"&&l[1]==="co"?!0:!1:!1;l&&(k=mc(k)||"",k=k.split("/"),k="/"+(k.length>1?k[1]:""),l=k==="/pagead");k=l?!0:!1}k=!k}k||e&&b[f]!==void 0||d==="TVHTML5_UNPLUGGED"&&h||(b[f]=g)}"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!lc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());if(c||!lc(a)){try{var m=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(n){}m&&
(b["X-YouTube-Time-Zone"]=m)}document.location.hostname.endsWith("youtubeeducation.com")||!c&&lc(a)||(b["X-YouTube-Ad-Signals"]=sq(yq()));return b}
function Sq(a,b){b.method="POST";b.postParams||(b.postParams={});return Tq(a,b)}
function Tq(a,b){var c=b.format||"JSON";a=Uq(a,b);var d=Vq(a,b),e=!1,f=Wq(a,function(k){if(!e){e=!0;h&&window.clearTimeout(h);var l=Cq(k),m=null,n=400<=k.status&&k.status<500,r=500<=k.status&&k.status<600;if(l||n||r)m=Xq(a,c,k,b.convertToSafeHtml);l&&(l=Yq(c,k,m));m=m||{};n=b.context||D;l?b.onSuccess&&b.onSuccess.call(n,k,m):b.onError&&b.onError.call(n,k,m);b.onFinish&&b.onFinish.call(n,k,m)}},b.method,d,b.headers,b.responseType,b.withCredentials,!1,b.onProgress);
d=b.timeout||0;if(b.onTimeout&&d>0){var g=b.onTimeout;var h=Eq(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||D,f))},d)}return f}
function Uq(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=S("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=vq(a,b);return a}
function Vq(a,b){var c=S("XSRF_FIELD_NAME"),d=S("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=S("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||lc(a)&&!b.withCredentials&&lc(a)!==document.location.hostname||b.method!=="POST"||h&&h!=="application/x-www-form-urlencoded"||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(T("ajax_parse_query_data_only_when_filled")&&f&&Object.keys(f).length>0||f)&&typeof e==="string"&&(e=tq(e),Pi(e,f),e=b.postBodyFormat&&b.postBodyFormat===
"JSON"?JSON.stringify(e):rc(e));f=e||f&&!Ii(f);!Oq&&f&&b.method!=="POST"&&(Oq=!0,lq(Error("AJAX request with postData should use POST")));return e}
function Xq(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,mq(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&a.indexOf("json")>=0&&(f.substring(0,5)===")]}'\n"&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?Zq(a):null)e={},Tb(a.getElementsByTagName("*"),function(g){e[g.tagName]=$q(g)})}d&&ar(e);
return e}
function ar(a){if(Pa(a))for(var b in a){var c;(c=b==="html_content")||(c=b.length-5,c=c>=0&&b.indexOf("_html",c)==c);if(c){c=a[b];var d=nb();c=d?d.createHTML(c):c;a[b]=new Fb(c)}else ar(a[b])}}
function Yq(a,b,c){if(b&&b.status===204)return!0;switch(a){case "JSON":return!!c;case "XML":return Number(c&&c.return_code)===0;case "RAW":return!0;default:return!!c}}
function Zq(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&a.length>0?a[0]:null:null}
function $q(a){var b="";Tb(a.childNodes,function(c){b+=c.nodeValue});
return b}
function Qq(a){var b=window.location.search,c=lc(a);T("debug_handle_relative_url_for_query_forward_killswitch")||!c&&xq(a)&&(c=document.location.hostname);var d=mc(a);d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=tq(b),f={};Tb(Nq,function(g){e[g]&&(f[g]=e[g])});
return wq(a,f||{},!1)}
var Wq=Pq;var br=[{ud:function(a){return"Cannot read property '"+a.key+"'"},
Sc:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{ud:function(a){return"Cannot call '"+a.key+"'"},
Sc:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{ud:function(a){return a.key+" is not defined"},
Sc:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var er={fb:[],Za:[{callback:cr,weight:500},{callback:dr,weight:500}]};function cr(a){if(a.name==="JavaException")return!0;a=a.stack;return a.includes("chrome://")||a.includes("-extension://")||a.includes("webkit-masked-url://")}
function dr(a){if(!a.stack)return!0;var b=!a.stack.includes("\n");return b&&a.stack.includes("ErrorType: ")||b&&a.stack.includes("Anonymous function (Unknown script")||a.stack.toLowerCase()==="not available"||a.fileName==="user-script"||a.fileName.startsWith("user-script:")?!0:!1}
;function fr(){this.Za=[];this.fb=[]}
var gr;function hr(){if(!gr){var a=gr=new fr;a.fb.length=0;a.Za.length=0;ir(a,er)}return gr}
function ir(a,b){b.fb&&a.fb.push.apply(a.fb,b.fb);b.Za&&a.Za.push.apply(a.Za,b.Za)}
;var jr=new O;function kr(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=lr(b);if(e===Infinity)break;var f=e>>3;switch(e&7){case 0:e=lr(b);if(f===2)return e;break;case 1:if(f===2)return;d+=8;break;case 2:e=lr(b);if(f===2)return a.substr(d,e);d+=e;break;case 5:if(f===2)return;d+=4;break;default:return}}while(d<c)}
function lr(a){var b=a(),c=b&127;if(b<128)return c;b=a();c|=(b&127)<<7;if(b<128)return c;b=a();c|=(b&127)<<14;if(b<128)return c;b=a();return b<128?c|(b&127)<<21:Infinity}
;function mr(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=nr(d,a[d],b,c),e>500));d++);d=e}else if(typeof a==="object")for(e in a){if(a[e]){var f=e;var g=a[e],h=b,k=c;f=typeof g!=="string"||f!=="clickTrackingParams"&&f!=="trackingParams"?0:(g=kr(atob(g.replace(/-/g,"+").replace(/_/g,"/"))))?nr(f+".ve",g,h,k):0;d+=f;d+=nr(e,a[e],b,c);if(d>500)break}}else c[b]=or(a),d+=c[b].length;else c[b]=or(a),d+=c[b].length;return d}
function nr(a,b,c,d){c+="."+a;a=or(b);d[c]=a;return c.length+a.length}
function or(a){try{return(typeof a==="string"?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;function pr(a){var b=this;this.i=void 0;this.h=!1;a.addEventListener("beforeinstallprompt",function(c){c.preventDefault();b.i=c});
a.addEventListener("appinstalled",function(){b.h=!0},{once:!0})}
function qr(){if(!D.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return D.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":D.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":D.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":D.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function rr(){this.ue=!0}
function sr(a){var b={},c=[];"USER_SESSION_ID"in fq&&c.push({key:"u",value:S("USER_SESSION_ID")});if(c=wi(c))b.Authorization=c,c=a=a==null?void 0:a.sessionIndex,c===void 0&&(c=Number(S("SESSION_INDEX",0)),c=isNaN(c)?0:c),T("voice_search_auth_header_removal")||(b["X-Goog-AuthUser"]=c.toString()),"INNERTUBE_HOST_OVERRIDE"in fq||(b["X-Origin"]=window.location.origin),a===void 0&&"DELEGATED_SESSION_ID"in fq&&(b["X-Goog-PageId"]=S("DELEGATED_SESSION_ID"));return b}
;var tr={identityType:"UNAUTHENTICATED_IDENTITY_TYPE_UNKNOWN"};function ur(a,b,c,d,e){ti.set(""+a,b,{kc:c,path:"/",domain:d===void 0?"youtube.com":d,secure:e===void 0?!1:e})}
function vr(a){return ti.get(""+a,void 0)}
function wr(a,b,c){ti.remove(""+a,b===void 0?"/":b,c===void 0?"youtube.com":c)}
function xr(){if(T("embeds_web_enable_cookie_detection_fix")){if(!D.navigator.cookieEnabled)return!1}else if(!ti.isEnabled())return!1;if(ti.h.cookie)return!0;T("embeds_web_enable_cookie_detection_fix")?ti.set("TESTCOOKIESENABLED","1",{kc:60,sameSite:"none",secure:!0}):ti.set("TESTCOOKIESENABLED","1",{kc:60});if(ti.get("TESTCOOKIESENABLED")!=="1")return!1;ti.remove("TESTCOOKIESENABLED");return!0}
;var yr=G("ytglobal.prefsUserPrefsPrefs_")||{};E("ytglobal.prefsUserPrefsPrefs_",yr);function zr(){this.h=S("ALT_PREF_COOKIE_NAME","PREF");this.i=S("ALT_PREF_COOKIE_DOMAIN","youtube.com");var a=vr(this.h);a&&this.parse(a)}
var Ar;function Br(){Ar||(Ar=new zr);return Ar}
p=zr.prototype;p.get=function(a,b){Cr(a);Dr(a);a=yr[a]!==void 0?yr[a].toString():null;return a!=null?a:b?b:""};
p.set=function(a,b){Cr(a);Dr(a);if(b==null)throw Error("ExpectedNotNull");yr[a]=b.toString()};
function Er(a){return!!((Fr("f"+(Math.floor(a/31)+1))||0)&1<<a%31)}
p.remove=function(a){Cr(a);Dr(a);delete yr[a]};
p.clear=function(){for(var a in yr)delete yr[a]};
function Dr(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function Cr(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function Fr(a){a=yr[a]!==void 0?yr[a].toString():null;return a!=null&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
p.parse=function(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(yr[d]=c.toString())}};var Gr={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},Hr={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};
function Ir(){var a=D.navigator;return a?a.connection:void 0}
function Jr(){var a=Ir();if(a){var b=Gr[a.type||"unknown"]||"CONN_UNKNOWN";a=Gr[a.effectiveType||"unknown"]||"CONN_UNKNOWN";b==="CONN_CELLULAR_UNKNOWN"&&a!=="CONN_UNKNOWN"&&(b=a);if(b!=="CONN_UNKNOWN")return b;if(a!=="CONN_UNKNOWN")return a}}
function Kr(){var a=Ir();if(a!=null&&a.effectiveType)return Hr.hasOwnProperty(a.effectiveType)?Hr[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN"}
;function U(a){var b=C.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(A(b));Object.setPrototypeOf(this,this.constructor.prototype)}
v(U,Error);function Lr(){try{return Mr(),!0}catch(a){return!1}}
function Mr(a){if(S("DATASYNC_ID")!==void 0)return S("DATASYNC_ID");throw new U("Datasync ID not set",a===void 0?"unknown":a);}
;function Nr(){}
function Or(a,b){return vn.Xa(a,0,b)}
Nr.prototype.ra=function(a,b){return this.Xa(a,1,b)};
Nr.prototype.Vb=function(a){var b=G("yt.scheduler.instance.addImmediateJob");b?b(a):a()};var Pr=Iq("web_emulated_idle_callback_delay",300),Qr=1E3/60-3,Rr=[8,5,4,3,2,1,0];
function Sr(a){a=a===void 0?{}:a;J.call(this);this.i=[];this.j={};this.ba=this.h=0;this.aa=this.u=!1;this.K=[];this.V=this.ha=!1;for(var b=y(Rr),c=b.next();!c.done;c=b.next())this.i[c.value]=[];this.o=0;this.gd=a.timeout||1;this.G=Qr;this.B=0;this.za=this.Vf.bind(this);this.Ub=this.Cg.bind(this);this.Va=this.Me.bind(this);this.Wa=this.zf.bind(this);this.lb=this.dg.bind(this);this.Ga=!!window.requestIdleCallback&&!!window.cancelIdleCallback&&!T("disable_scheduler_requestIdleCallback");(this.na=a.useRaf!==
!1&&!!window.requestAnimationFrame)&&document.addEventListener("visibilitychange",this.za)}
v(Sr,J);p=Sr.prototype;p.Vb=function(a){var b=Za();Tr(this,a);a=Za()-b;this.u||(this.G-=a)};
p.Xa=function(a,b,c){++this.ba;if(b===10)return this.Vb(a),this.ba;var d=this.ba;this.j[d]=a;this.u&&!c?this.K.push({id:d,priority:b}):(this.i[b].push(d),this.aa||this.u||(this.h!==0&&Ur(this)!==this.B&&this.stop(),this.start()));return d};
p.sa=function(a){delete this.j[a]};
function Vr(a){a.K.length=0;for(var b=5;b>=0;b--)a.i[b].length=0;a.i[8].length=0;a.j={};a.stop()}
p.isHidden=function(){return!!document.hidden||!1};
function Wr(a){return!a.isHidden()&&a.na}
function Ur(a){if(a.i[8].length){if(a.V)return 4;if(Wr(a))return 3}for(var b=5;b>=a.o;b--)if(a.i[b].length>0)return b>0?Wr(a)?3:2:1;return 0}
p.wa=function(a){var b=G("yt.logging.errors.log");b&&b(a)};
function Tr(a,b){try{b()}catch(c){a.wa(c)}}
function Xr(a){for(var b=y(Rr),c=b.next();!c.done;c=b.next())if(a.i[c.value].length)return!0;return!1}
p.zf=function(a){var b=void 0;a&&(b=a.timeRemaining());this.ha=!0;Yr(this,b);this.ha=!1};
p.Cg=function(){Yr(this)};
p.Me=function(){Zr(this)};
p.dg=function(a){this.V=!0;var b=Ur(this);b===4&&b!==this.B&&(this.stop(),this.start());Yr(this,void 0,a);this.V=!1};
p.Vf=function(){this.isHidden()||Zr(this);this.h&&(this.stop(),this.start())};
function Zr(a){a.stop();a.u=!0;for(var b=Za(),c=a.i[8];c.length;){var d=c.shift(),e=a.j[d];delete a.j[d];e&&Tr(a,e)}$r(a);a.u=!1;Xr(a)&&a.start();b=Za()-b;a.G-=b}
function $r(a){for(var b=0,c=a.K.length;b<c;b++){var d=a.K[b];a.i[d.priority].push(d.id)}a.K.length=0}
function Yr(a,b,c){a.V&&a.B===4&&a.h||a.stop();a.u=!0;b=Za()+(b||a.G);for(var d=a.i[5];d.length;){var e=d.shift(),f=a.j[e];delete a.j[e];if(f){e=a;try{f(c)}catch(l){e.wa(l)}}}for(d=a.i[4];d.length;)c=d.shift(),f=a.j[c],delete a.j[c],f&&Tr(a,f);d=a.ha?0:1;d=a.o>d?a.o:d;if(!(Za()>=b)){do{a:{c=a;f=d;for(e=3;e>=f;e--)for(var g=c.i[e];g.length;){var h=g.shift(),k=c.j[h];delete c.j[h];if(k){c=k;break a}}c=null}c&&Tr(a,c)}while(c&&Za()<b)}a.u=!1;$r(a);a.G=Qr;Xr(a)&&a.start()}
p.start=function(){this.aa=!1;if(this.h===0)switch(this.B=Ur(this),this.B){case 1:var a=this.Wa;this.h=this.Ga?window.requestIdleCallback(a,{timeout:3E3}):window.setTimeout(a,Pr);break;case 2:this.h=window.setTimeout(this.Ub,this.gd);break;case 3:this.h=window.requestAnimationFrame(this.lb);break;case 4:this.h=window.setTimeout(this.Va,0)}};
p.pause=function(){this.stop();this.aa=!0};
p.stop=function(){if(this.h){switch(this.B){case 1:var a=this.h;this.Ga?window.cancelIdleCallback(a):window.clearTimeout(a);break;case 2:case 4:window.clearTimeout(this.h);break;case 3:window.cancelAnimationFrame(this.h)}this.h=0}};
p.X=function(){Vr(this);this.stop();this.na&&document.removeEventListener("visibilitychange",this.za);J.prototype.X.call(this)};var as=G("yt.scheduler.instance.timerIdMap_")||{},bs=Iq("kevlar_tuner_scheduler_soft_state_timer_ms",800),cs=0,ds=0;function es(){var a=G("ytglobal.schedulerInstanceInstance_");if(!a||a.I)a=new Sr(S("scheduler")||{}),E("ytglobal.schedulerInstanceInstance_",a);return a}
function gs(){hs();var a=G("ytglobal.schedulerInstanceInstance_");a&&(yc(a),E("ytglobal.schedulerInstanceInstance_",null))}
function hs(){Vr(es());for(var a in as)as.hasOwnProperty(a)&&delete as[Number(a)]}
function is(a,b,c){if(!c)return c=c===void 0,-es().Xa(a,b,c);var d=window.setTimeout(function(){var e=es().Xa(a,b);as[d]=e},c);
return d}
function js(a){es().Vb(a)}
function ks(a){var b=es();if(a<0)b.sa(-a);else{var c=as[a];c?(b.sa(c),delete as[a]):window.clearTimeout(a)}}
function ls(){ms()}
function ms(){window.clearTimeout(cs);es().start()}
function ns(){es().pause();window.clearTimeout(cs);cs=window.setTimeout(ls,bs)}
function ps(){window.clearTimeout(ds);ds=window.setTimeout(function(){qs(0)},bs)}
function qs(a){ps();var b=es();b.o=a;b.start()}
function rs(a){ps();var b=es();b.o>a&&(b.o=a,b.start())}
function ss(){window.clearTimeout(ds);var a=es();a.o=0;a.start()}
;function ts(){Nr.apply(this,arguments)}
v(ts,Nr);function us(){ts.instance||(ts.instance=new ts);return ts.instance}
ts.prototype.Xa=function(a,b,c){c!==void 0&&Number.isNaN(Number(c))&&(c=void 0);var d=G("yt.scheduler.instance.addJob");return d?d(a,b,c):c===void 0?(a(),NaN):Eq(a,c||0)};
ts.prototype.sa=function(a){if(a===void 0||!Number.isNaN(Number(a))){var b=G("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
ts.prototype.start=function(){var a=G("yt.scheduler.instance.start");a&&a()};
ts.prototype.pause=function(){var a=G("yt.scheduler.instance.pause");a&&a()};
var vn=us();
G("yt.scheduler.initialized")||(E("yt.scheduler.instance.dispose",gs),E("yt.scheduler.instance.addJob",is),E("yt.scheduler.instance.addImmediateJob",js),E("yt.scheduler.instance.cancelJob",ks),E("yt.scheduler.instance.cancelAllJobs",hs),E("yt.scheduler.instance.start",ms),E("yt.scheduler.instance.pause",ns),E("yt.scheduler.instance.setPriorityThreshold",qs),E("yt.scheduler.instance.enablePriorityThreshold",rs),E("yt.scheduler.instance.clearPriorityThreshold",ss),E("yt.scheduler.initialized",!0));function vs(a){var b=new Vn;this.h=(a=b.isAvailable()?a?new Wn(b,a):b:null)?new Qn(a):null;this.i=document.domain||window.location.hostname}
vs.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+c*1E3);return}catch(f){}var e="";if(d)try{e=escape((new fl).serialize(b))}catch(f){return}else e=escape(b);ur(a,e,c,this.i)};
vs.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=vr(a))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
vs.prototype.remove=function(a){this.h&&this.h.remove(a);wr(a,"/",this.i)};var ws=function(){var a;return function(){a||(a=new vs("ytidb"));return a}}();
function xs(){var a;return(a=ws())==null?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var ys=[],zs,As=!1;function Bs(){var a={};for(zs=new Cs(a.handleError===void 0?Ds:a.handleError,a.logEvent===void 0?Es:a.logEvent);ys.length>0;)switch(a=ys.shift(),a.type){case "ERROR":zs.wa(a.payload);break;case "EVENT":zs.logEvent(a.eventType,a.payload)}}
function Fs(a){As||(zs?zs.wa(a):(ys.push({type:"ERROR",payload:a}),ys.length>10&&ys.shift()))}
function Gs(a,b){As||(zs?zs.logEvent(a,b):(ys.push({type:"EVENT",eventType:a,payload:b}),ys.length>10&&ys.shift()))}
;function Hs(a){if(a.indexOf(":")>=0)throw Error("Database name cannot contain ':'");}
function Is(a){return a.substr(0,a.indexOf(":"))||a}
;var Js=Ad||Bd;function Ks(a){var b=jd();return b?b.toLowerCase().indexOf(a)>=0:!1}
;var Ls={},Ms=(Ls.AUTH_INVALID="No user identifier specified.",Ls.EXPLICIT_ABORT="Transaction was explicitly aborted.",Ls.IDB_NOT_SUPPORTED="IndexedDB is not supported.",Ls.MISSING_INDEX="Index not created.",Ls.MISSING_OBJECT_STORES="Object stores not created.",Ls.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",Ls.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",Ls.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
Ls.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",Ls.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",Ls.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",Ls.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",Ls),Ns={},Os=(Ns.AUTH_INVALID="ERROR",Ns.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",Ns.EXPLICIT_ABORT="IGNORED",Ns.IDB_NOT_SUPPORTED="ERROR",Ns.MISSING_INDEX=
"WARNING",Ns.MISSING_OBJECT_STORES="ERROR",Ns.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",Ns.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",Ns.QUOTA_EXCEEDED="WARNING",Ns.QUOTA_MAYBE_EXCEEDED="WARNING",Ns.UNKNOWN_ABORT="WARNING",Ns.INCOMPATIBLE_DB_VERSION="WARNING",Ns),Ps={},Qs=(Ps.AUTH_INVALID=!1,Ps.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,Ps.EXPLICIT_ABORT=!1,Ps.IDB_NOT_SUPPORTED=!1,Ps.MISSING_INDEX=!1,Ps.MISSING_OBJECT_STORES=!1,Ps.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,Ps.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,Ps.QUOTA_EXCEEDED=!1,Ps.QUOTA_MAYBE_EXCEEDED=!0,Ps.UNKNOWN_ABORT=!0,Ps.INCOMPATIBLE_DB_VERSION=!1,Ps);function Rs(a,b,c,d,e){b=b===void 0?{}:b;c=c===void 0?Ms[a]:c;d=d===void 0?Os[a]:d;e=e===void 0?Qs[a]:e;U.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:self.document===void 0,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,Rs.prototype)}
v(Rs,U);function Ss(a,b){Rs.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},Ms.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,Ss.prototype)}
v(Ss,Rs);function Ts(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,Ts.prototype)}
v(Ts,Error);var Us=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function Vs(a,b,c,d){b=Is(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof Rs)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if(e.name==="QuotaExceededError")return new Rs("QUOTA_EXCEEDED",a);if(Cd&&e.name==="UnknownError")return new Rs("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof Ts)return new Rs("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if(e.name==="InvalidStateError"&&Us.some(function(f){return e.message.includes(f)}))return new Rs("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if(e.name==="AbortError")return new Rs("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",ee:e.name})];e.level="WARNING";return e}
function Ws(a,b,c){var d=xs();return new Rs("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:d==null?void 0:d.hasSucceededOnce}})}
;function Xs(a){if(!a)throw Error();throw a;}
function Ys(a){return a}
function Zs(a){this.h=a}
function $s(a){function b(e){if(d.state.status==="PENDING"){d.state={status:"REJECTED",reason:e};e=y(d.i);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if(d.state.status==="PENDING"){d.state={status:"FULFILLED",value:e};e=y(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.h=[];this.i=[];a=a.h;try{a(c,b)}catch(e){b(e)}}
$s.all=function(a){return new $s(new Zs(function(b,c){var d=[],e=a.length;e===0&&b(d);for(var f={Hb:0};f.Hb<a.length;f={Hb:f.Hb},++f.Hb)$s.resolve(a[f.Hb]).then(function(g){return function(h){d[g.Hb]=h;e--;e===0&&b(d)}}(f)).catch(function(g){c(g)})}))};
$s.resolve=function(a){return new $s(new Zs(function(b,c){a instanceof $s?a.then(b,c):b(a)}))};
$s.reject=function(a){return new $s(new Zs(function(b,c){c(a)}))};
$s.prototype.then=function(a,b){var c=this,d=a!=null?a:Ys,e=b!=null?b:Xs;return new $s(new Zs(function(f,g){c.state.status==="PENDING"?(c.h.push(function(){at(c,c,d,f,g)}),c.i.push(function(){bt(c,c,e,f,g)})):c.state.status==="FULFILLED"?at(c,c,d,f,g):c.state.status==="REJECTED"&&bt(c,c,e,f,g)}))};
$s.prototype.catch=function(a){return this.then(void 0,a)};
function at(a,b,c,d,e){try{if(a.state.status!=="FULFILLED")throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof $s?ct(a,b,f,d,e):d(f)}catch(g){e(g)}}
function bt(a,b,c,d,e){try{if(a.state.status!=="REJECTED")throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof $s?ct(a,b,f,d,e):d(f)}catch(g){e(g)}}
function ct(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof $s?ct(a,b,f,d,e):d(f)},function(f){e(f)})}
;function dt(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function et(a){return new Promise(function(b,c){dt(a,b,c)})}
function ft(a){return new $s(new Zs(function(b,c){dt(a,b,c)}))}
;function gt(a,b){return new $s(new Zs(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var ht=window,W=ht.ytcsi&&ht.ytcsi.now?ht.ytcsi.now:ht.performance&&ht.performance.timing&&ht.performance.now&&ht.performance.timing.navigationStart?function(){return ht.performance.timing.navigationStart+ht.performance.now()}:function(){return(new Date).getTime()};function jt(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(W());this.i=!1}
function kt(){return T("idb_immediate_commit")}
p=jt.prototype;p.add=function(a,b,c){return lt(this,[a],{mode:"readwrite",ka:!0,commit:kt()},function(d){return d.objectStore(a).add(b,c)})};
p.clear=function(a){return lt(this,[a],{mode:"readwrite",ka:!0},function(b){return b.objectStore(a).clear()})};
p.close=function(){this.h.close();var a;((a=this.options)==null?0:a.closed)&&this.options.closed()};
p.count=function(a,b){return lt(this,[a],{mode:"readonly",ka:!0,commit:kt()},function(c){return c.objectStore(a).count(b)})};
function mt(a,b,c){a=a.h.createObjectStore(b,c);return new nt(a)}
p.delete=function(a,b){return lt(this,[a],{mode:"readwrite",ka:!0,commit:kt()&&!(b instanceof IDBKeyRange)},function(c){return c.objectStore(a).delete(b)})};
p.get=function(a,b){return lt(this,[a],{mode:"readonly",ka:!0,commit:kt()},function(c){return c.objectStore(a).get(b)})};
function ot(a,b,c){return lt(a,[b],{mode:"readwrite",ka:!0,commit:kt()},function(d){d=d.objectStore(b);return ft(d.h.put(c,void 0))})}
p.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function lt(a,b,c,d){var e,f,g,h,k,l,m,n,r,t,w,z;return B(function(x){switch(x.h){case 1:var H={mode:"readonly",ka:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};typeof c==="string"?H.mode=c:Object.assign(H,c);e=H;a.transactionCount++;f=e.ka?3:1;g=0;case 2:if(h){x.v(4);break}g++;k=Math.round(W());wa(x,5);l=a.h.transaction(b,e.mode);H=x.yield;var F=!!e.commit;var V=new pt(l);F=qt(V,d,F);return H.call(x,F,7);case 7:return m=x.i,n=Math.round(W()),rt(a,k,n,g,void 0,b.join(),e),x.return(m);case 5:r=ya(x);t=Math.round(W());
w=Vs(r,a.h.name,b.join(),a.h.version);if((z=w instanceof Rs&&!w.h)||g>=f)rt(a,k,t,g,w,b.join(),e),h=w;x.v(2);break;case 4:return x.return(Promise.reject(h))}})}
function rt(a,b,c,d,e,f,g){b=c-b;e?(e instanceof Rs&&(e.type==="QUOTA_EXCEEDED"||e.type==="QUOTA_MAYBE_EXCEEDED")&&Gs("QUOTA_EXCEEDED",{dbName:Is(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof Rs&&e.type==="UNKNOWN_ABORT"&&(c-=a.j,c<0&&c>=2147483648&&(c=0),Gs("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.i=!0),st(a,!1,d,f,b,g.tag),Fs(e)):st(a,!0,d,f,b,g.tag)}
function st(a,b,c,d,e,f){Gs("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c,tag:f===void 0?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
p.getName=function(){return this.h.name};
function nt(a){this.h=a}
p=nt.prototype;p.add=function(a,b){return ft(this.h.add(a,b))};
p.autoIncrement=function(){return this.h.autoIncrement};
p.clear=function(){return ft(this.h.clear()).then(function(){})};
function tt(a,b,c){a.h.createIndex(b,c,{unique:!1})}
p.count=function(a){return ft(this.h.count(a))};
function ut(a,b){return vt(a,{query:b},function(c){return c.delete().then(function(){return wt(c)})}).then(function(){})}
p.delete=function(a){return a instanceof IDBKeyRange?ut(this,a):ft(this.h.delete(a))};
p.get=function(a){return ft(this.h.get(a))};
p.index=function(a){try{return new xt(this.h.index(a))}catch(b){if(b instanceof Error&&b.name==="NotFoundError")throw new Ts(a,this.h.name);throw b;}};
p.getName=function(){return this.h.name};
p.keyPath=function(){return this.h.keyPath};
function vt(a,b,c){a=a.h.openCursor(b.query,b.direction);return zt(a).then(function(d){return gt(d,c)})}
function pt(a){var b=this;this.h=a;this.i=new Map;this.aborted=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.aborted){e=Rs;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var k=f.item(h);if(k===null)throw Error("Invariant: item in DOMStringList is null");g.push(k)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function qt(a,b,c){var d=new Promise(function(e,f){try{var g=b(a);c&&a.commit();g.then(function(h){e(h)}).catch(f)}catch(h){f(h),a.abort()}});
return Promise.all([d,a.done]).then(function(e){return y(e).next().value})}
pt.prototype.abort=function(){this.h.abort();this.aborted=!0;throw new Rs("EXPLICIT_ABORT");};
pt.prototype.commit=function(){if(!this.aborted){var a,b;(b=(a=this.h).commit)==null||b.call(a)}};
pt.prototype.objectStore=function(a){a=this.h.objectStore(a);var b=this.i.get(a);b||(b=new nt(a),this.i.set(a,b));return b};
function xt(a){this.h=a}
p=xt.prototype;p.count=function(a){return ft(this.h.count(a))};
p.delete=function(a){return At(this,{query:a},function(b){return b.delete().then(function(){return wt(b)})})};
p.get=function(a){return ft(this.h.get(a))};
p.keyPath=function(){return this.h.keyPath};
p.unique=function(){return this.h.unique};
function At(a,b,c){a=a.h.openCursor(b.query===void 0?null:b.query,b.direction===void 0?"next":b.direction);return zt(a).then(function(d){return gt(d,c)})}
function Bt(a,b){this.request=a;this.cursor=b}
function zt(a){return ft(a).then(function(b){return b?new Bt(a,b):null})}
function wt(a){a.cursor.continue(void 0);return zt(a.request)}
Bt.prototype.delete=function(){return ft(this.cursor.delete()).then(function(){})};
Bt.prototype.getValue=function(){return this.cursor.value};
Bt.prototype.update=function(a){return ft(this.cursor.update(a))};function Ct(a,b,c){return new Promise(function(d,e){function f(){r||(r=new jt(g.result,{closed:n}));return r}
var g=b!==void 0?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.Qe,k=c.blocking,l=c.Ag,m=c.upgrade,n=c.closed,r;g.addEventListener("upgradeneeded",function(t){try{if(t.newVersion===null)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(g.transaction===null)throw Error("Invariant: transaction on IDbOpenDbRequest is null");t.dataLoss&&t.dataLoss!=="none"&&Gs("IDB_DATA_CORRUPTED",{reason:t.dataLossMessage||"unknown reason",dbName:Is(a)});var w=f(),z=new pt(g.transaction);
m&&m(w,function(x){return t.oldVersion<x&&t.newVersion>=x},z);
z.done.catch(function(x){e(x)})}catch(x){e(x)}});
g.addEventListener("success",function(){var t=g.result;k&&t.addEventListener("versionchange",function(){k(f())});
t.addEventListener("close",function(){Gs("IDB_UNEXPECTEDLY_CLOSED",{dbName:Is(a),dbVersion:t.version});l&&l()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function Dt(a,b,c){c=c===void 0?{}:c;return Ct(a,b,c)}
function Et(a,b){b=b===void 0?{}:b;var c,d,e,f;return B(function(g){if(g.h==1)return wa(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.Qe)&&c.addEventListener("blocked",function(){e()}),g.yield(et(c),4);
if(g.h!=2)return xa(g,0);f=ya(g);throw Vs(f,a,"",-1);})}
;function Ft(a,b){this.name=a;this.options=b;this.j=!0;this.u=this.o=0}
Ft.prototype.i=function(a,b,c){c=c===void 0?{}:c;return Dt(a,b,c)};
Ft.prototype.delete=function(a){a=a===void 0?{}:a;return Et(this.name,a)};
function Gt(a,b){return new Rs("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function Ht(a,b){if(!b)throw Ws("openWithToken",Is(a.name));return a.open()}
Ft.prototype.open=function(){function a(){var f,g,h,k,l,m,n,r,t,w;return B(function(z){switch(z.h){case 1:return g=(f=Error().stack)!=null?f:"",wa(z,2),z.yield(c.i(c.name,c.options.version,e),4);case 4:for(var x=h=z.i,H=c.options,F=[],V=y(Object.keys(H.Nb)),na=V.next();!na.done;na=V.next()){na=na.value;var ka=H.Nb[na],Od=ka.eg===void 0?Number.MAX_VALUE:ka.eg;!(x.h.version>=ka.Xb)||x.h.version>=Od||x.h.objectStoreNames.contains(na)||F.push(na)}k=F;if(k.length===0){z.v(5);break}l=Object.keys(c.options.Nb);
m=h.objectStoreNames();if(c.u<Iq("ytidb_reopen_db_retries",0))return c.u++,h.close(),Fs(new Rs("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),z.return(a());if(!(c.o<Iq("ytidb_remake_db_retries",1))){z.v(6);break}c.o++;return z.yield(c.delete(),7);case 7:return Fs(new Rs("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:c.name,expectedObjectStores:l,foundObjectStores:m})),z.return(a());case 6:throw new Ss(m,l);case 5:return z.return(h);case 2:n=ya(z);
if(n instanceof DOMException?n.name!=="VersionError":"DOMError"in self&&n instanceof DOMError?n.name!=="VersionError":!(n instanceof Object&&"message"in n)||n.message!=="An attempt was made to open a database using a lower version than the existing version."){z.v(8);break}return z.yield(c.i(c.name,void 0,Object.assign({},e,{upgrade:void 0})),9);case 9:r=z.i;t=r.h.version;if(c.options.version!==void 0&&t>c.options.version+1)throw r.close(),c.j=!1,Gt(c,t);return z.return(r);case 8:throw b(),n instanceof
Error&&!T("ytidb_async_stack_killswitch")&&(n.stack=n.stack+"\n"+g.substring(g.indexOf("\n")+1)),Vs(n,c.name,"",(w=c.options.version)!=null?w:-1);}})}
function b(){c.h===d&&(c.h=void 0)}
var c=this;if(!this.j)throw Gt(this);if(this.h)return this.h;var d,e={blocking:function(f){f.close()},
closed:b,Ag:b,upgrade:this.options.upgrade};return this.h=d=a()};var It=new Ft("YtIdbMeta",{Nb:{databases:{Xb:1}},upgrade:function(a,b){b(1)&&mt(a,"databases",{keyPath:"actualName"})}});
function Jt(a,b){var c;return B(function(d){if(d.h==1)return d.yield(Ht(It,b),2);c=d.i;return d.return(lt(c,["databases"],{ka:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return ft(f.h.put(a,void 0)).then(function(){})})}))})}
function Kt(a,b){var c;return B(function(d){if(d.h==1)return a?d.yield(Ht(It,b),2):d.return();c=d.i;return d.return(c.delete("databases",a))})}
function Lt(a,b){var c,d;return B(function(e){return e.h==1?(c=[],e.yield(Ht(It,b),2)):e.h!=3?(d=e.i,e.yield(lt(d,["databases"],{ka:!0,mode:"readonly"},function(f){c.length=0;return vt(f.objectStore("databases"),{},function(g){a(g.getValue())&&c.push(g.getValue());return wt(g)})}),3)):e.return(c)})}
function Mt(a){return Lt(function(b){return b.publicName==="LogsDatabaseV2"&&b.userIdentifier!==void 0},a)}
function Nt(a,b,c){return Lt(function(d){return c?d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)&&c.includes(d.publicName):d.userIdentifier!==void 0&&!a.includes(d.userIdentifier)},b)}
function Ot(a){var b,c;return B(function(d){if(d.h==1)return b=Mr("YtIdbMeta hasAnyMeta other"),d.yield(Lt(function(e){return e.userIdentifier!==void 0&&e.userIdentifier!==b},a),2);
c=d.i;return d.return(c.length>0)})}
;var Pt,Qt=new function(){}(new function(){});
function Rt(){var a,b,c,d;return B(function(e){switch(e.h){case 1:a=xs();if((b=a)==null?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=Js)f=/WebKit\/([0-9]+)/.exec(jd()),f=!!(f&&parseInt(f[1],10)>=600);f&&(f=/WebKit\/([0-9]+)/.exec(jd()),f=!(f&&parseInt(f[1],10)>=602));if(f||wd)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
wa(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return e.yield(Jt(d,Qt),4);case 4:return e.yield(Kt("yt-idb-test-do-not-use",Qt),5);case 5:return e.return(!0);case 2:return ya(e),e.return(!1)}})}
function St(){if(Pt!==void 0)return Pt;As=!0;return Pt=Rt().then(function(a){As=!1;var b;if((b=ws())!=null&&b.h){var c;b={hasSucceededOnce:((c=xs())==null?void 0:c.hasSucceededOnce)||a};var d;(d=ws())==null||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function Tt(){return G("ytglobal.idbToken_")||void 0}
function Ut(){var a=Tt();return a?Promise.resolve(a):St().then(function(b){(b=b?Qt:void 0)&&E("ytglobal.idbToken_",b);return b})}
;var Vt=0;function Wt(a,b){Vt||(Vt=vn.ra(function(){var c,d,e,f,g;return B(function(h){switch(h.h){case 1:return h.yield(Ut(),2);case 2:c=h.i;if(!c)return h.return();d=!0;wa(h,3);return h.yield(Nt(a,c,b),5);case 5:e=h.i;if(!e.length){d=!1;h.v(6);break}f=e[0];return h.yield(Et(f.actualName),7);case 7:return h.yield(Kt(f.actualName,c),6);case 6:xa(h,4);break;case 3:g=ya(h),Fs(g),d=!1;case 4:vn.sa(Vt),Vt=0,d&&Wt(a,b),h.h=0}})}))}
function Xt(){var a;return B(function(b){return b.h==1?b.yield(Ut(),2):(a=b.i)?b.return(Ot(a)):b.return(!1)})}
new Ml;function Yt(a){if(!Lr())throw a=new Rs("AUTH_INVALID",{dbName:a}),Fs(a),a;var b=Mr();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function Zt(a,b,c,d){var e,f,g,h,k,l;return B(function(m){switch(m.h){case 1:return f=(e=Error().stack)!=null?e:"",m.yield(Ut(),2);case 2:g=m.i;if(!g)throw h=Ws("openDbImpl",a,b),T("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),Fs(h),h;Hs(a);k=c?{actualName:a,publicName:a,userIdentifier:void 0}:Yt(a);wa(m,3);return m.yield(Jt(k,g),5);case 5:return m.yield(Dt(k.actualName,b,d),6);case 6:return m.return(m.i);case 3:return l=ya(m),wa(m,7),m.yield(Kt(k.actualName,
g),9);case 9:xa(m,8);break;case 7:ya(m);case 8:throw l;}})}
function $t(a,b,c){c=c===void 0?{}:c;return Zt(a,b,!1,c)}
function au(a,b,c){c=c===void 0?{}:c;return Zt(a,b,!0,c)}
function bu(a,b){b=b===void 0?{}:b;var c,d;return B(function(e){if(e.h==1)return e.yield(Ut(),2);if(e.h!=3){c=e.i;if(!c)return e.return();Hs(a);d=Yt(a);return e.yield(Et(d.actualName,b),3)}return e.yield(Kt(d.actualName,c),0)})}
function cu(a,b,c){a=a.map(function(d){return B(function(e){return e.h==1?e.yield(Et(d.actualName,b),2):e.yield(Kt(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function du(){var a=a===void 0?{}:a;var b,c;return B(function(d){if(d.h==1)return d.yield(Ut(),2);if(d.h!=3){b=d.i;if(!b)return d.return();Hs("LogsDatabaseV2");return d.yield(Mt(b),3)}c=d.i;return d.yield(cu(c,a,b),0)})}
function eu(a,b){b=b===void 0?{}:b;var c;return B(function(d){if(d.h==1)return d.yield(Ut(),2);if(d.h!=3){c=d.i;if(!c)return d.return();Hs(a);return d.yield(Et(a,b),3)}return d.yield(Kt(a,c),0)})}
;function fu(a,b){Ft.call(this,a,b);this.options=b;Hs(a)}
v(fu,Ft);function gu(a,b){var c;return function(){c||(c=new fu(a,b));return c}}
fu.prototype.i=function(a,b,c){c=c===void 0?{}:c;return(this.options.shared?au:$t)(a,b,Object.assign({},c))};
fu.prototype.delete=function(a){a=a===void 0?{}:a;return(this.options.shared?eu:bu)(this.name,a)};
function hu(a,b){return gu(a,b)}
;var iu={},ju=hu("ytGcfConfig",{Nb:(iu.coldConfigStore={Xb:1},iu.hotConfigStore={Xb:1},iu),shared:!1,upgrade:function(a,b){b(1)&&(tt(mt(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),tt(mt(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function ku(a){return Ht(ju(),a)}
function lu(a,b,c){var d,e,f;return B(function(g){switch(g.h){case 1:return d={config:a,hashData:b,timestamp:W()},g.yield(ku(c),2);case 2:return e=g.i,g.yield(e.clear("hotConfigStore"),3);case 3:return g.yield(ot(e,"hotConfigStore",d),4);case 4:return f=g.i,g.return(f)}})}
function mu(a,b,c,d){var e,f,g;return B(function(h){switch(h.h){case 1:return e={config:a,hashData:b,configData:c,timestamp:W()},h.yield(ku(d),2);case 2:return f=h.i,h.yield(f.clear("coldConfigStore"),3);case 3:return h.yield(ot(f,"coldConfigStore",e),4);case 4:return g=h.i,h.return(g)}})}
function nu(a){var b,c;return B(function(d){return d.h==1?d.yield(ku(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(lt(b,["coldConfigStore"],{mode:"readwrite",ka:!0},function(e){return At(e.objectStore("coldConfigStore").index("coldTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
function ou(a){var b,c;return B(function(d){return d.h==1?d.yield(ku(a),2):d.h!=3?(b=d.i,c=void 0,d.yield(lt(b,["hotConfigStore"],{mode:"readwrite",ka:!0},function(e){return At(e.objectStore("hotConfigStore").index("hotTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
;function pu(){J.call(this);this.i=[];this.h=[];var a=G("yt.gcf.config.hotUpdateCallbacks");a?(this.i=[].concat(A(a)),this.h=a):(this.h=[],E("yt.gcf.config.hotUpdateCallbacks",this.h))}
v(pu,J);pu.prototype.X=function(){for(var a=y(this.i),b=a.next();!b.done;b=a.next()){var c=this.h;b=c.indexOf(b.value);b>=0&&c.splice(b,1)}this.i.length=0;J.prototype.X.call(this)};function qu(){this.h=0;this.i=new pu}
function ru(){var a;return(a=G("yt.gcf.config.hotConfigGroup"))!=null?a:S("RAW_HOT_CONFIG_GROUP")}
function su(a,b,c){var d,e,f;return B(function(g){switch(g.h){case 1:if(!T("start_client_gcf")){g.v(0);break}c&&(a.j=c,E("yt.gcf.config.hotConfigGroup",a.j||null));a.o(b);d=Tt();if(!d){g.v(3);break}if(c){g.v(4);break}return g.yield(ou(d),5);case 5:e=g.i,c=(f=e)==null?void 0:f.config;case 4:return g.yield(lu(c,b,d),3);case 3:if(c)for(var h=c,k=y(a.i.h),l=k.next();!l.done;l=k.next())l=l.value,l(h);g.h=0}})}
function tu(a,b,c){var d,e,f,g;return B(function(h){if(h.h==1){if(!T("start_client_gcf"))return h.v(0);a.coldHashData=b;E("yt.gcf.config.coldHashData",a.coldHashData||null);return(d=Tt())?c?h.v(4):h.yield(nu(d),5):h.v(0)}h.h!=4&&(e=h.i,c=(f=e)==null?void 0:f.config);if(!c)return h.v(0);g=c.configData;return h.yield(mu(c,b,g,d),0)})}
function uu(){if(!qu.instance){var a=new qu;qu.instance=a}a=qu.instance;var b=W()-a.h;if(!(a.h!==0&&b<Iq("send_config_hash_timer"))){b=G("yt.gcf.config.coldConfigData");var c=G("yt.gcf.config.hotHashData"),d=G("yt.gcf.config.coldHashData");b&&c&&d&&(a.h=W());return{coldConfigData:b,hotHashData:c,coldHashData:d}}}
qu.prototype.o=function(a){this.hotHashData=a;E("yt.gcf.config.hotHashData",this.hotHashData||null)};function vu(){return"INNERTUBE_API_KEY"in fq&&"INNERTUBE_API_VERSION"in fq}
function wu(){return{innertubeApiKey:S("INNERTUBE_API_KEY"),innertubeApiVersion:S("INNERTUBE_API_VERSION"),Bf:S("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),Xd:S("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),dj:S("INNERTUBE_CONTEXT_CLIENT_NAME",1),innertubeContextClientVersion:S("INNERTUBE_CONTEXT_CLIENT_VERSION"),Df:S("INNERTUBE_CONTEXT_HL"),Cf:S("INNERTUBE_CONTEXT_GL"),Ef:S("INNERTUBE_HOST_OVERRIDE")||"",Ff:!!S("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),ej:!!S("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:S("SERIALIZED_CLIENT_CONFIG_DATA")}}
function xu(a){var b={client:{hl:a.Df,gl:a.Cf,clientName:a.Xd,clientVersion:a.innertubeContextClientVersion,configInfo:a.Bf}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=D.devicePixelRatio;c&&c!=1&&(b.client.screenDensityFloat=String(c));c=S("EXPERIMENTS_TOKEN","");c!==""&&(b.client.experimentsToken=c);c=Lq();c.length>0&&(b.request={internalExperimentFlags:c});c=a.Xd;if((c==="WEB"||c==="MWEB"||c===1||c===2)&&b){var d;b.client.mainAppWebInfo=(d=b.client.mainAppWebInfo)!=
null?d:{};b.client.mainAppWebInfo.webDisplayMode=qr()}(d=G("yt.embedded_player.embed_url"))&&b&&(b.thirdParty={embedUrl:d});var e;if(T("web_log_memory_total_kbytes")&&((e=D.navigator)==null?0:e.deviceMemory)){var f;e=(f=D.navigator)==null?void 0:f.deviceMemory;b&&(b.client.memoryTotalKbytes=""+e*1E6)}a.appInstallData&&b&&(b.client.configInfo=b.client.configInfo||{},b.client.configInfo.appInstallData=a.appInstallData);(a=Jr())&&b&&(b.client.connectionType=a);T("web_log_effective_connection_type")&&
(a=Kr())&&b&&(b.client.effectiveConnectionType=a);T("start_client_gcf")&&(e=uu())&&(a=e.coldConfigData,f=e.coldHashData,e=e.hotHashData,b&&(b.client.configInfo=b.client.configInfo||{},a&&(b.client.configInfo.coldConfigData=a),f&&(b.client.configInfo.coldHashData=f),e&&(b.client.configInfo.hotHashData=e)));S("DELEGATED_SESSION_ID")&&!T("pageid_as_header_web")&&(b.user={onBehalfOfUser:S("DELEGATED_SESSION_ID")});!T("fill_delegate_context_in_gel_killswitch")&&(a=S("INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT"))&&
(b.user=Object.assign({},b.user,{serializedDelegationContext:a}));a=S("INNERTUBE_CONTEXT");var g;if(T("enable_persistent_device_token")&&(a==null?0:(g=a.client)==null?0:g.rolloutToken)){var h;b.client.rolloutToken=a==null?void 0:(h=a.client)==null?void 0:h.rolloutToken}g=Object;h=g.assign;a=b.client;f={};e=y(Object.entries(tq(S("DEVICE",""))));for(d=e.next();!d.done;d=e.next())c=y(d.value),d=c.next().value,c=c.next().value,d==="cbrand"?f.deviceMake=c:d==="cmodel"?f.deviceModel=c:d==="cbr"?f.browserName=
c:d==="cbrver"?f.browserVersion=c:d==="cos"?f.osName=c:d==="cosver"?f.osVersion=c:d==="cplatform"&&(f.platform=c);b.client=h.call(g,a,f);return b}
function yu(a,b,c){c=c===void 0?{}:c;var d={};S("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":S("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||S("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;b=c.authorization||S("AUTHORIZATION");b||(a?b="Bearer "+G("gapi.auth.getToken")().Si:(rr.instance||(rr.instance=new rr),a=sr(),T("pageid_as_header_web")||delete a["X-Goog-PageId"],d=Object.assign({},d,a)));b&&(d.Authorization=b);return d}
;var zu=typeof TextEncoder!=="undefined"?new TextEncoder:null,Au=zu?function(a){return zu.encode(a)}:function(a){a=fd(a);
for(var b=new Uint8Array(a.length),c=0;c<b.length;c++)b[c]=a[c];return b};var Bu={next:"wn_s",browse:"br_s",search:"sr_s",reel:"r_wrs",player:"ps_s"},Cu={next:"wn_r",browse:"br_r",search:"sr_r",reel:"r_wrr",player:"ps_r"};function Du(a,b){this.version=a;this.args=b}
Du.prototype.serialize=function(){return{version:this.version,args:this.args}};function Eu(a,b){this.topic=a;this.h=b}
Eu.prototype.toString=function(){return this.topic};var Fu=G("ytPubsub2Pubsub2Instance")||new O;O.prototype.subscribe=O.prototype.subscribe;O.prototype.unsubscribeByKey=O.prototype.xc;O.prototype.publish=O.prototype.yb;O.prototype.clear=O.prototype.clear;E("ytPubsub2Pubsub2Instance",Fu);var Gu=G("ytPubsub2Pubsub2SubscribedKeys")||{};E("ytPubsub2Pubsub2SubscribedKeys",Gu);var Hu=G("ytPubsub2Pubsub2TopicToKeys")||{};E("ytPubsub2Pubsub2TopicToKeys",Hu);var Iu=G("ytPubsub2Pubsub2IsAsync")||{};E("ytPubsub2Pubsub2IsAsync",Iu);
E("ytPubsub2Pubsub2SkipSubKey",null);function Ju(a,b){var c=Ku();c&&c.publish.call(c,a.toString(),a,b)}
function Lu(a){var b=Mu,c=Ku();if(!c)return 0;var d=c.subscribe(b.toString(),function(e,f){var g=G("ytPubsub2Pubsub2SkipSubKey");g&&g==d||(g=function(){if(Gu[d])try{if(f&&b instanceof Eu&&b!=e)try{var h=b.h,k=f;if(!k.args||!k.version)throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");try{if(!h.Be){var l=new h;h.Be=l.version}var m=h.Be}catch(n){}if(!m||k.version!=m)throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");try{f=Reflect.construct(h,
Zb(k.args))}catch(n){throw n.message="yt.pubsub2.Data.deserialize(): "+n.message,n;}}catch(n){throw n.message="yt.pubsub2.pubsub2 cross-binary conversion error for "+b.toString()+": "+n.message,n;}a.call(window,f)}catch(n){lq(n)}},Iu[b.toString()]?G("yt.scheduler.instance")?vn.ra(g):Eq(g,0):g())});
Gu[d]=!0;Hu[b.toString()]||(Hu[b.toString()]=[]);Hu[b.toString()].push(d);return d}
function Nu(){var a=Ou,b=Lu(function(c){a.apply(void 0,arguments);Pu(b)});
return b}
function Pu(a){var b=Ku();b&&(typeof a==="number"&&(a=[a]),Tb(a,function(c){b.unsubscribeByKey(c);delete Gu[c]}))}
function Ku(){return G("ytPubsub2Pubsub2Instance")}
;function Qu(a,b,c){c=c===void 0?{sampleRate:.1}:c;Math.random()<Math.min(.02,c.sampleRate/100)&&Ju("meta_logging_csi_event",{timerName:a,Gj:b})}
;var Ru=void 0,Su=void 0;function Tu(){Su||(Su=Ep(S("WORKER_SERIALIZATION_URL")));return Su||void 0}
function Uu(){var a=Tu();Ru||a===void 0||(Ru=new Worker(qb(a),void 0));return Ru}
;var Vu=Iq("max_body_size_to_compress",5E5),Wu=Iq("min_body_size_to_compress",500),Xu=!0,Yu=0,Zu=0,$u=Iq("compression_performance_threshold_lr",250),av=Iq("slow_compressions_before_abandon_count",4),bv=!1,cv=new Map,dv=1;function ev(){if(typeof Worker==="function"&&Tu()&&!bv){var a=function(c){c=c.data;if(c.op==="gzippedGelBatch"){var d=cv.get(c.key);d&&(fv(c.gzippedBatch,d.latencyPayload,d.url,d.options,d.sendFn),cv.delete(c.key))}},b=Uu();
b&&(b.addEventListener("message",a),b.onerror=function(){cv.clear()},bv=!0)}}
function gv(a,b,c,d,e){e=e===void 0?!1:e;var f={startTime:W(),ticks:{},infos:{}};if(Xu)try{var g=hv(b);if(g!=null&&(g>Vu||g<Wu))d(a,c);else{if(T("gzip_gel_with_worker")){bv||ev();var h=Uu();if(h&&!e){cv.set(dv,{latencyPayload:f,url:a,options:c,sendFn:d});h.postMessage({op:"gelBatchToGzip",serializedBatch:b,key:dv});dv++;return}}var k=Cp(Au(b));fv(k,f,a,c,d)}}catch(l){mq(l),d(a,c)}else d(a,c)}
function fv(a,b,c,d,e){var f=W();b.ticks.gelc=f;Zu++;T("disable_compression_due_to_performance_degredation")&&f-b.startTime>=$u&&(Yu++,Xu=!1);iv(b);d.headers||(d.headers={});d.headers["Content-Encoding"]="gzip";d.postBody=a;d.postParams=void 0;e(c,d)}
function jv(a){var b=b===void 0?!1:b;var c=c===void 0?!1:c;var d=W(),e={startTime:d,ticks:{},infos:{}},f=b?G("yt.logging.gzipForFetch",!1):!0;if(Xu&&f){if(!a.body)return a;try{var g=c?a.body:typeof a.body==="string"?a.body:JSON.stringify(a.body);f=g;if(!c&&typeof g==="string"){var h=hv(g);if(h!=null&&(h>Vu||h<Wu))return a;c=b?{level:1}:void 0;f=Cp(Au(g),c);var k=W();e.ticks.gelc=k;if(b){Zu++;if((T("disable_compression_due_to_performance_degredation")||T("disable_compression_due_to_performance_degradation_lr"))&&
k-d>=$u)if(Yu++,T("abandon_compression_after_N_slow_zips_lr")){b=Yu/Zu;var l=av/Iq("compression_disable_point");Zu>0&&Zu%Iq("compression_disable_point")===0&&b>=l&&(Xu=!1)}else Xu=!1;iv(e)}}a.headers=Object.assign({},{"Content-Encoding":"gzip"},a.headers||{});a.body=f;return a}catch(m){return mq(m),a}}else return a}
function hv(a){try{return(new Blob(a.split(""))).size}catch(b){return mq(b),null}}
function iv(a){T("gel_compression_csi_killswitch")||!T("log_gel_compression_latency")&&!T("log_gel_compression_latency_lr")||Qu("gel_compression",a,{sampleRate:.1})}
;function kv(a){a=Object.assign({},a);delete a.Authorization;var b=wi();if(b){var c=new zn;c.update(S("INNERTUBE_API_KEY"));c.update(b);a.hash=Fd(c.digest(),3)}return a}
;var lv;function mv(){lv||(lv=new vs("yt.innertube"));return lv}
function nv(a,b,c,d){if(d)return null;d=mv().get("nextId",!0)||1;var e=mv().get("requests",!0)||{};e[d]={method:a,request:b,authState:kv(c),requestTime:Math.round(W())};mv().set("nextId",d+1,86400,!0);mv().set("requests",e,86400,!0);return d}
function ov(a){var b=mv().get("requests",!0)||{};delete b[a];mv().set("requests",b,86400,!0)}
function pv(a){var b=mv().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(Math.round(W())-d.requestTime<6E4)){var e=d.authState,f=kv(yu(!1));Li(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(W())),qv(a,d.method,e,{}));delete b[c]}}mv().set("requests",b,86400,!0)}}
;function rv(a){this.zc=this.h=!1;this.potentialEsfErrorCounter=this.i=0;this.handleError=function(){};
this.Fb=function(){};
this.now=Date.now;this.dc=!1;var b;this.we=(b=a.we)!=null?b:100;var c;this.pe=(c=a.pe)!=null?c:1;var d;this.ne=(d=a.ne)!=null?d:2592E6;var e;this.he=(e=a.he)!=null?e:12E4;var f;this.oe=(f=a.oe)!=null?f:5E3;var g;this.W=(g=a.W)!=null?g:void 0;this.Gc=!!a.Gc;var h;this.Ec=(h=a.Ec)!=null?h:.1;var k;this.Uc=(k=a.Uc)!=null?k:10;a.handleError&&(this.handleError=a.handleError);a.Fb&&(this.Fb=a.Fb);a.dc&&(this.dc=a.dc);a.zc&&(this.zc=a.zc);this.Y=a.Y;this.Ea=a.Ea;this.ga=a.ga;this.fa=a.fa;this.sendFn=a.sendFn;
this.Ad=a.Ad;this.xd=a.xd;sv(this)&&(!this.Y||this.Y("networkless_logging"))&&tv(this)}
function tv(a){sv(a)&&!a.dc&&(a.h=!0,a.Gc&&Math.random()<=a.Ec&&a.ga.Ue(a.W),uv(a),a.fa.va()&&a.wc(),a.fa.listen(a.Ad,a.wc.bind(a)),a.fa.listen(a.xd,a.Pd.bind(a)))}
p=rv.prototype;p.writeThenSend=function(a,b){var c=this;b=b===void 0?{}:b;if(sv(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.ga.set(d,this.W).then(function(e){d.id=e;c.fa.va()&&vv(c,d)}).catch(function(e){vv(c,d);
wv(c,e)})}else this.sendFn(a,b)};
p.sendThenWrite=function(a,b,c){var d=this;b=b===void 0?{}:b;if(sv(this)&&this.h){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.Y&&this.Y("nwl_skip_retry")&&(e.skipRetry=c);if(this.fa.va()||this.Y&&this.Y("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return B(function(k){if(k.h==1)return k.yield(d.ga.set(e,d.W).catch(function(l){wv(d,l)}),2);
f(g,h);k.h=0})}}this.sendFn(a,b,e.skipRetry)}else this.ga.set(e,this.W).catch(function(g){d.sendFn(a,b,e.skipRetry);
wv(d,g)})}else this.sendFn(a,b,this.Y&&this.Y("nwl_skip_retry")&&c)};
p.sendAndWrite=function(a,b){var c=this;b=b===void 0?{}:b;if(sv(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){d.id!==void 0?c.ga.Cb(d.id,c.W):e=!0;c.fa.sb&&c.Y&&c.Y("vss_network_hint")&&c.fa.sb(!0);f(g,h)};
this.sendFn(d.url,d.options,void 0,!0);this.ga.set(d,this.W).then(function(g){d.id=g;e&&c.ga.Cb(d.id,c.W)}).catch(function(g){wv(c,g)})}else this.sendFn(a,b,void 0,!0)};
p.wc=function(){var a=this;if(!sv(this))throw Error("IndexedDB is not supported: throttleSend");this.i||(this.i=this.Ea.ra(function(){var b;return B(function(c){if(c.h==1)return c.yield(a.ga.Td("NEW",a.W),2);if(c.h!=3)return b=c.i,b?c.yield(vv(a,b),3):(a.Pd(),c.return());a.i&&(a.i=0,a.wc());c.h=0})},this.we))};
p.Pd=function(){this.Ea.sa(this.i);this.i=0};
function vv(a,b){var c;return B(function(d){switch(d.h){case 1:if(!sv(a))throw Error("IndexedDB is not supported: immediateSend");if(b.id===void 0){d.v(2);break}return d.yield(a.ga.Lf(b.id,a.W),3);case 3:(c=d.i)||a.Fb(Error("The request cannot be found in the database."));case 2:if(xv(a,b,a.ne)){d.v(4);break}a.Fb(Error("Networkless Logging: Stored logs request expired age limit"));if(b.id===void 0){d.v(5);break}return d.yield(a.ga.Cb(b.id,a.W),5);case 5:return d.return();case 4:b.skipRetry||(b=yv(a,
b));if(!b){d.v(0);break}if(!b.skipRetry||b.id===void 0){d.v(8);break}return d.yield(a.ga.Cb(b.id,a.W),8);case 8:a.sendFn(b.url,b.options,!!b.skipRetry),d.h=0}})}
function yv(a,b){if(!sv(a))throw Error("IndexedDB is not supported: updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,k,l;return B(function(m){switch(m.h){case 1:g=zv(f);(h=Av(f))&&a.Y&&a.Y("web_enable_error_204")&&a.handleError(Error("Request failed due to compression"),b.url,f);if(!(a.Y&&a.Y("nwl_consider_error_code")&&g||a.Y&&!a.Y("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.Uc)){m.v(2);break}if(!a.fa.Yc){m.v(3);break}return m.yield(a.fa.Yc(),3);case 3:if(a.fa.va()){m.v(2);break}c(e,f);if(!a.Y||!a.Y("nwl_consider_error_code")||((k=b)==null?void 0:k.id)===void 0){m.v(6);
break}return m.yield(a.ga.Bd(b.id,a.W,!1),6);case 6:return m.return();case 2:if(a.Y&&a.Y("nwl_consider_error_code")&&!g&&a.potentialEsfErrorCounter>a.Uc)return m.return();a.potentialEsfErrorCounter++;if(((l=b)==null?void 0:l.id)===void 0){m.v(8);break}return b.sendCount<a.pe?m.yield(a.ga.Bd(b.id,a.W,!0,h?!1:void 0),12):m.yield(a.ga.Cb(b.id,a.W),8);case 12:a.Ea.ra(function(){a.fa.va()&&a.wc()},a.oe);
case 8:c(e,f),m.h=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return B(function(h){if(h.h==1)return((g=b)==null?void 0:g.id)===void 0?h.v(2):h.yield(a.ga.Cb(b.id,a.W),2);a.fa.sb&&a.Y&&a.Y("vss_network_hint")&&a.fa.sb(!0);d(e,f);h.h=0})};
return b}
function xv(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function uv(a){if(!sv(a))throw Error("IndexedDB is not supported: retryQueuedRequests");a.ga.Td("QUEUED",a.W).then(function(b){b&&!xv(a,b,a.he)?a.Ea.ra(function(){return B(function(c){if(c.h==1)return b.id===void 0?c.v(2):c.yield(a.ga.Bd(b.id,a.W),2);uv(a);c.h=0})}):a.fa.va()&&a.wc()})}
function wv(a,b){a.He&&!a.fa.va()?a.He(b):a.handleError(b)}
function sv(a){return!!a.W||a.zc}
function zv(a){var b;return(a=a==null?void 0:(b=a.error)==null?void 0:b.code)&&a>=400&&a<=599?!1:!0}
function Av(a){var b;a=a==null?void 0:(b=a.error)==null?void 0:b.code;return!(a!==400&&a!==415)}
;var Bv;
function Cv(){if(Bv)return Bv();var a={};Bv=hu("LogsDatabaseV2",{Nb:(a.LogsRequestsStore={Xb:2},a),shared:!1,upgrade:function(b,c,d){c(2)&&mt(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.h.indexNames.contains("newRequest")&&d.h.deleteIndex("newRequest"),tt(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.h.objectStoreNames.contains("sapisid")&&b.h.deleteObjectStore("sapisid");c(9)&&b.h.objectStoreNames.contains("SWHealthLog")&&b.h.deleteObjectStore("SWHealthLog")},
version:9});return Bv()}
;function Hv(a){return Ht(Cv(),a)}
function Iv(a,b){var c,d,e,f;return B(function(g){if(g.h==1)return c={startTime:W(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},ticks:{}},g.yield(Hv(b),2);if(g.h!=3)return d=g.i,e=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:S("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),g.yield(ot(d,"LogsRequestsStore",e),3);f=g.i;c.ticks.tc=W();Jv(c);return g.return(f)})}
function Kv(a,b){var c,d,e,f,g,h,k,l,m;return B(function(n){if(n.h==1)return c={startTime:W(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},ticks:{}},n.yield(Hv(b),2);if(n.h!=3)return d=n.i,e=S("INNERTUBE_CONTEXT_CLIENT_NAME",0),f=[a,e,0],g=[a,e,W()],h=IDBKeyRange.bound(f,g),k="prev",T("use_fifo_for_networkless")&&(k="next"),l=void 0,m=a==="NEW"?"readwrite":"readonly",T("use_readonly_for_get_most_recent_by_status_killswitch")&&(m="readwrite"),n.yield(lt(d,["LogsRequestsStore"],{mode:m,ka:!0},
function(r){return At(r.objectStore("LogsRequestsStore").index("newRequestV2"),{query:h,direction:k},function(t){t.getValue()&&(l=t.getValue(),a==="NEW"&&(l.status="QUEUED",t.update(l)))})}),3);
c.ticks.tc=W();Jv(c);return n.return(l)})}
function Lv(a,b){var c;return B(function(d){if(d.h==1)return d.yield(Hv(b),2);c=d.i;return d.return(lt(c,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",ft(f.h.put(g,void 0)).then(function(){return g})})}))})}
function Mv(a,b,c,d){c=c===void 0?!0:c;var e;return B(function(f){if(f.h==1)return f.yield(Hv(b),2);e=f.i;return f.return(lt(e,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(g){var h=g.objectStore("LogsRequestsStore");return h.get(a).then(function(k){return k?(k.status="NEW",c&&(k.sendCount+=1),d!==void 0&&(k.options.compress=d),ft(h.h.put(k,void 0)).then(function(){return k})):$s.resolve(void 0)})}))})}
function Nv(a,b){var c;return B(function(d){if(d.h==1)return d.yield(Hv(b),2);c=d.i;return d.return(c.delete("LogsRequestsStore",a))})}
function Ov(a){var b,c;return B(function(d){if(d.h==1)return d.yield(Hv(a),2);b=d.i;c=W()-2592E6;return d.yield(lt(b,["LogsRequestsStore"],{mode:"readwrite",ka:!0},function(e){return vt(e.objectStore("LogsRequestsStore"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return wt(f)})})}),0)})}
function Pv(){B(function(a){return a.yield(du(),0)})}
function Jv(a){T("nwl_csi_killswitch")||Qu("networkless_performance",a,{sampleRate:1})}
;var Qv={accountStateChangeSignedIn:23,accountStateChangeSignedOut:24,delayedEventMetricCaptured:11,latencyActionBaselined:6,latencyActionInfo:7,latencyActionTicked:5,offlineTransferStatusChanged:2,offlineImageDownload:335,playbackStartStateChanged:9,systemHealthCaptured:3,mangoOnboardingCompleted:10,mangoPushNotificationReceived:230,mangoUnforkDbMigrationError:121,mangoUnforkDbMigrationSummary:122,mangoUnforkDbMigrationPreunforkDbVersionNumber:133,mangoUnforkDbMigrationPhoneMetadata:134,mangoUnforkDbMigrationPhoneStorage:135,
mangoUnforkDbMigrationStep:142,mangoAsyncApiMigrationEvent:223,mangoDownloadVideoResult:224,mangoHomepageVideoCount:279,mangoHomeV3State:295,mangoImageClientCacheHitEvent:273,sdCardStatusChanged:98,framesDropped:12,thumbnailHovered:13,deviceRetentionInfoCaptured:14,thumbnailLoaded:15,backToAppEvent:318,streamingStatsCaptured:17,offlineVideoShared:19,appCrashed:20,youThere:21,offlineStateSnapshot:22,mdxSessionStarted:25,mdxSessionConnected:26,mdxSessionDisconnected:27,bedrockResourceConsumptionSnapshot:28,
nextGenWatchWatchSwiped:29,kidsAccountsSnapshot:30,zeroStepChannelCreated:31,tvhtml5SearchCompleted:32,offlineSharePairing:34,offlineShareUnlock:35,mdxRouteDistributionSnapshot:36,bedrockRepetitiveActionTimed:37,unpluggedDegradationInfo:229,uploadMp4HeaderMoved:38,uploadVideoTranscoded:39,uploadProcessorStarted:46,uploadProcessorEnded:47,uploadProcessorReady:94,uploadProcessorRequirementPending:95,uploadProcessorInterrupted:96,uploadFrontendEvent:241,assetPackDownloadStarted:41,assetPackDownloaded:42,
assetPackApplied:43,assetPackDeleted:44,appInstallAttributionEvent:459,playbackSessionStopped:45,adBlockerMessagingShown:48,distributionChannelCaptured:49,dataPlanCpidRequested:51,detailedNetworkTypeCaptured:52,sendStateUpdated:53,receiveStateUpdated:54,sendDebugStateUpdated:55,receiveDebugStateUpdated:56,kidsErrored:57,mdxMsnSessionStatsFinished:58,appSettingsCaptured:59,mdxWebSocketServerHttpError:60,mdxWebSocketServer:61,startupCrashesDetected:62,coldStartInfo:435,offlinePlaybackStarted:63,liveChatMessageSent:225,
liveChatUserPresent:434,liveChatBeingModerated:457,liveCreationCameraUpdated:64,liveCreationEncodingCaptured:65,liveCreationError:66,liveCreationHealthUpdated:67,liveCreationVideoEffectsCaptured:68,liveCreationStageOccured:75,liveCreationBroadcastScheduled:123,liveCreationArchiveReplacement:149,liveCreationCostreamingConnection:421,liveCreationPlayablesMetrics:533,liveCreationStreamWebrtcStats:288,liveCreationWebrtcError:526,mdxSessionRecoveryStarted:69,mdxSessionRecoveryCompleted:70,mdxSessionRecoveryStopped:71,
visualElementShown:72,visualElementHidden:73,visualElementGestured:78,visualElementStateChanged:208,screenCreated:156,playbackAssociated:202,visualElementAttached:215,playbackContextEvent:214,cloudCastingPlaybackStarted:74,webPlayerApiCalled:76,tvhtml5AccountDialogOpened:79,foregroundHeartbeat:80,foregroundHeartbeatScreenAssociated:111,kidsOfflineSnapshot:81,mdxEncryptionSessionStatsFinished:82,playerRequestCompleted:83,liteSchedulerStatistics:84,mdxSignIn:85,spacecastMetadataLookupRequested:86,spacecastBatchLookupRequested:87,
spacecastSummaryRequested:88,spacecastPlayback:89,spacecastDiscovery:90,tvhtml5LaunchUrlComponentChanged:91,mdxBackgroundPlaybackRequestCompleted:92,mdxBrokenAdditionalDataDeviceDetected:93,tvhtml5LocalStorage:97,tvhtml5DeviceStorageStatus:147,autoCaptionsAvailable:99,playbackScrubbingEvent:339,flexyState:100,interfaceOrientationCaptured:101,mainAppBrowseFragmentCache:102,offlineCacheVerificationFailure:103,offlinePlaybackExceptionDigest:217,vrCopresenceStats:104,vrCopresenceSyncStats:130,vrCopresenceCommsStats:137,
vrCopresencePartyStats:153,vrCopresenceEmojiStats:213,vrCopresenceEvent:141,vrCopresenceFlowTransitEvent:160,vrCowatchPartyEvent:492,vrCowatchUserStartOrJoinEvent:504,vrPlaybackEvent:345,kidsAgeGateTracking:105,offlineDelayAllowedTracking:106,mainAppAutoOfflineState:107,videoAsThumbnailDownload:108,videoAsThumbnailPlayback:109,liteShowMore:110,renderingError:118,kidsProfilePinGateTracking:119,abrTrajectory:124,scrollEvent:125,streamzIncremented:126,kidsProfileSwitcherTracking:127,kidsProfileCreationTracking:129,
buyFlowStarted:136,mbsConnectionInitiated:138,mbsPlaybackInitiated:139,mbsLoadChildren:140,liteProfileFetcher:144,mdxRemoteTransaction:146,reelPlaybackError:148,reachabilityDetectionEvent:150,mobilePlaybackEvent:151,courtsidePlayerStateChanged:152,musicPersistentCacheChecked:154,musicPersistentCacheCleared:155,playbackInterrupted:157,playbackInterruptionResolved:158,fixFopFlow:159,anrDetection:161,backstagePostCreationFlowEnded:162,clientError:163,gamingAccountLinkStatusChanged:164,liteHousewarming:165,
buyFlowEvent:167,kidsParentalGateTracking:168,kidsSignedOutSettingsStatus:437,kidsSignedOutPauseHistoryFixStatus:438,tvhtml5WatchdogViolation:444,ypcUpgradeFlow:169,yongleStudy:170,ypcUpdateFlowStarted:171,ypcUpdateFlowCancelled:172,ypcUpdateFlowSucceeded:173,ypcUpdateFlowFailed:174,liteGrowthkitPromo:175,paymentFlowStarted:341,transactionFlowShowPaymentDialog:405,transactionFlowStarted:176,transactionFlowSecondaryDeviceStarted:222,transactionFlowSecondaryDeviceSignedOutStarted:383,transactionFlowCancelled:177,
transactionFlowPaymentCallBackReceived:387,transactionFlowPaymentSubmitted:460,transactionFlowPaymentSucceeded:329,transactionFlowSucceeded:178,transactionFlowFailed:179,transactionFlowPlayBillingConnectionStartEvent:428,transactionFlowSecondaryDeviceSuccess:458,transactionFlowErrorEvent:411,liteVideoQualityChanged:180,watchBreakEnablementSettingEvent:181,watchBreakFrequencySettingEvent:182,videoEffectsCameraPerformanceMetrics:183,adNotify:184,startupTelemetry:185,playbackOfflineFallbackUsed:186,
outOfMemory:187,ypcPauseFlowStarted:188,ypcPauseFlowCancelled:189,ypcPauseFlowSucceeded:190,ypcPauseFlowFailed:191,uploadFileSelected:192,ypcResumeFlowStarted:193,ypcResumeFlowCancelled:194,ypcResumeFlowSucceeded:195,ypcResumeFlowFailed:196,adsClientStateChange:197,ypcCancelFlowStarted:198,ypcCancelFlowCancelled:199,ypcCancelFlowSucceeded:200,ypcCancelFlowFailed:201,ypcCancelFlowGoToPaymentProcessor:402,ypcDeactivateFlowStarted:320,ypcRedeemFlowStarted:203,ypcRedeemFlowCancelled:204,ypcRedeemFlowSucceeded:205,
ypcRedeemFlowFailed:206,ypcFamilyCreateFlowStarted:258,ypcFamilyCreateFlowCancelled:259,ypcFamilyCreateFlowSucceeded:260,ypcFamilyCreateFlowFailed:261,ypcFamilyManageFlowStarted:262,ypcFamilyManageFlowCancelled:263,ypcFamilyManageFlowSucceeded:264,ypcFamilyManageFlowFailed:265,restoreContextEvent:207,embedsAdEvent:327,autoplayTriggered:209,clientDataErrorEvent:210,experimentalVssValidation:211,tvhtml5TriggeredEvent:212,tvhtml5FrameworksFieldTrialResult:216,tvhtml5FrameworksFieldTrialStart:220,musicOfflinePreferences:218,
watchTimeSegment:219,appWidthLayoutError:221,accountRegistryChange:226,userMentionAutoCompleteBoxEvent:227,downloadRecommendationEnablementSettingEvent:228,musicPlaybackContentModeChangeEvent:231,offlineDbOpenCompleted:232,kidsFlowEvent:233,kidsFlowCorpusSelectedEvent:234,videoEffectsEvent:235,unpluggedOpsEogAnalyticsEvent:236,playbackAudioRouteEvent:237,interactionLoggingDebugModeError:238,offlineYtbRefreshed:239,kidsFlowError:240,musicAutoplayOnLaunchAttempted:242,deviceContextActivityEvent:243,
deviceContextEvent:244,templateResolutionException:245,musicSideloadedPlaylistServiceCalled:246,embedsStorageAccessNotChecked:247,embedsHasStorageAccessResult:248,embedsItpPlayedOnReload:249,embedsRequestStorageAccessResult:250,embedsShouldRequestStorageAccessResult:251,embedsRequestStorageAccessState:256,embedsRequestStorageAccessFailedState:257,embedsItpWatchLaterResult:266,searchSuggestDecodingPayloadFailure:252,siriShortcutActivated:253,tvhtml5KeyboardPerformance:254,latencyActionSpan:255,elementsLog:267,
ytbFileOpened:268,tfliteModelError:269,apiTest:270,yongleUsbSetup:271,touStrikeInterstitialEvent:272,liteStreamToSave:274,appBundleClientEvent:275,ytbFileCreationFailed:276,adNotifyFailure:278,ytbTransferFailed:280,blockingRequestFailed:281,liteAccountSelector:282,liteAccountUiCallbacks:283,dummyPayload:284,browseResponseValidationEvent:285,entitiesError:286,musicIosBackgroundFetch:287,mdxNotificationEvent:289,layersValidationError:290,musicPwaInstalled:291,liteAccountCleanup:292,html5PlayerHealthEvent:293,
watchRestoreAttempt:294,liteAccountSignIn:296,notaireEvent:298,kidsVoiceSearchEvent:299,adNotifyFilled:300,delayedEventDropped:301,analyticsSearchEvent:302,systemDarkThemeOptOutEvent:303,flowEvent:304,networkConnectivityBaselineEvent:305,ytbFileImported:306,downloadStreamUrlExpired:307,directSignInEvent:308,lyricImpressionEvent:309,accessibilityStateEvent:310,tokenRefreshEvent:311,genericAttestationExecution:312,tvhtml5VideoSeek:313,unpluggedAutoPause:314,scrubbingEvent:315,bedtimeReminderEvent:317,
tvhtml5UnexpectedRestart:319,tvhtml5DeviceStorageStats:535,tvhtml5StabilityTraceEvent:478,tvhtml5OperationHealth:467,tvhtml5WatchKeyEvent:321,voiceLanguageChanged:322,tvhtml5LiveChatStatus:323,parentToolsCorpusSelectedEvent:324,offerAdsEnrollmentInitiated:325,networkQualityIntervalEvent:326,deviceStartupMetrics:328,heartbeatActionPlayerTransitioned:330,tvhtml5Lifecycle:331,heartbeatActionPlayerHalted:332,adaptiveInlineMutedSettingEvent:333,mainAppLibraryLoadingState:334,thirdPartyLogMonitoringEvent:336,
appShellAssetLoadReport:337,tvhtml5AndroidAttestation:338,tvhtml5StartupSoundEvent:340,iosBackgroundRefreshTask:342,iosBackgroundProcessingTask:343,sliEventBatch:344,postImpressionEvent:346,musicSideloadedPlaylistExport:347,idbUnexpectedlyClosed:348,voiceSearchEvent:349,mdxSessionCastEvent:350,idbQuotaExceeded:351,idbTransactionEnded:352,idbTransactionAborted:353,tvhtml5KeyboardLogging:354,idbIsSupportedCompleted:355,creatorStudioMobileEvent:356,idbDataCorrupted:357,parentToolsAppChosenEvent:358,
webViewBottomSheetResized:359,activeStateControllerScrollPerformanceSummary:360,navigatorValidation:361,mdxSessionHeartbeat:362,clientHintsPolyfillDiagnostics:363,clientHintsPolyfillEvent:364,proofOfOriginTokenError:365,kidsAddedAccountSummary:366,musicWearableDevice:367,ypcRefundFlowEvent:368,tvhtml5PlaybackMeasurementEvent:369,tvhtml5WatermarkMeasurementEvent:370,clientExpGcfPropagationEvent:371,mainAppReferrerIntent:372,leaderLockEnded:373,leaderLockAcquired:374,googleHatsEvent:375,persistentLensLaunchEvent:376,
parentToolsChildWelcomeChosenEvent:378,browseThumbnailPreloadEvent:379,finalPayload:380,mdxDialAdditionalDataUpdateEvent:381,webOrchestrationTaskLifecycleRecord:382,startupSignalEvent:384,accountError:385,gmsDeviceCheckEvent:386,accountSelectorEvent:388,accountUiCallbacks:389,mdxDialAdditionalDataProbeEvent:390,downloadsSearchIcingApiStats:391,downloadsSearchIndexUpdatedEvent:397,downloadsSearchIndexSnapshot:398,dataPushClientEvent:392,kidsCategorySelectedEvent:393,mdxDeviceManagementSnapshotEvent:394,
prefetchRequested:395,prefetchableCommandExecuted:396,gelDebuggingEvent:399,webLinkTtsPlayEnd:400,clipViewInvalid:401,persistentStorageStateChecked:403,cacheWipeoutEvent:404,playerEvent:410,sfvEffectPipelineStartedEvent:412,sfvEffectPipelinePausedEvent:429,sfvEffectPipelineEndedEvent:413,sfvEffectChosenEvent:414,sfvEffectLoadedEvent:415,sfvEffectUserInteractionEvent:465,sfvEffectFirstFrameProcessedLatencyEvent:416,sfvEffectAggregatedFramesProcessedLatencyEvent:417,sfvEffectAggregatedFramesDroppedEvent:418,
sfvEffectPipelineErrorEvent:430,sfvEffectGraphFrozenEvent:419,sfvEffectGlThreadBlockedEvent:420,mdeQosEvent:510,mdeVideoChangedEvent:442,mdePlayerPerformanceMetrics:472,mdeExporterEvent:497,genericClientExperimentEvent:423,homePreloadTaskScheduled:424,homePreloadTaskExecuted:425,homePreloadCacheHit:426,polymerPropertyChangedInObserver:427,applicationStarted:431,networkCronetRttBatch:432,networkCronetRttSummary:433,repeatChapterLoopEvent:436,seekCancellationEvent:462,lockModeTimeoutEvent:483,externalVideoShareToYoutubeAttempt:501,
parentCodeEvent:502,offlineTransferStarted:4,musicOfflineMixtapePreferencesChanged:16,mangoDailyNewVideosNotificationAttempt:40,mangoDailyNewVideosNotificationError:77,dtwsPlaybackStarted:112,dtwsTileFetchStarted:113,dtwsTileFetchCompleted:114,dtwsTileFetchStatusChanged:145,dtwsKeyframeDecoderBufferSent:115,dtwsTileUnderflowedOnNonkeyframe:116,dtwsBackfillFetchStatusChanged:143,dtwsBackfillUnderflowed:117,dtwsAdaptiveLevelChanged:128,blockingVisitorIdTimeout:277,liteSocial:18,mobileJsInvocation:297,
biscottiBasedDetection:439,coWatchStateChange:440,embedsVideoDataDidChange:441,shortsFirst:443,cruiseControlEvent:445,qoeClientLoggingContext:446,atvRecommendationJobExecuted:447,tvhtml5UserFeedback:448,producerProjectCreated:449,producerProjectOpened:450,producerProjectDeleted:451,producerProjectElementAdded:453,producerProjectElementRemoved:454,producerAppStateChange:509,producerProjectDiskInsufficientExportFailure:516,producerMediaServicesResetDetails:522,tvhtml5ShowClockEvent:455,deviceCapabilityCheckMetrics:456,
youtubeClearcutEvent:461,offlineBrowseFallbackEvent:463,getCtvTokenEvent:464,startupDroppedFramesSummary:466,screenshotEvent:468,miniAppPlayEvent:469,elementsDebugCounters:470,fontLoadEvent:471,webKillswitchReceived:473,webKillswitchExecuted:474,cameraOpenEvent:475,manualSmoothnessMeasurement:476,tvhtml5AppQualityEvent:477,polymerPropertyAccessEvent:479,miniAppSdkUsage:480,cobaltTelemetryEvent:481,crossDevicePlayback:482,channelCreatedWithObakeImage:484,channelEditedWithObakeImage:485,offlineDeleteEvent:486,
crossDeviceNotificationTransfer:487,androidIntentEvent:488,unpluggedAmbientInterludesCounterfactualEvent:489,keyPlaysPlayback:490,shortsCreationFallbackEvent:493,vssData:491,castMatch:494,miniAppPerformanceMetrics:495,userFeedbackEvent:496,kidsGuestSessionMismatch:498,musicSideloadedPlaylistMigrationEvent:499,sleepTimerSessionFinishEvent:500,watchEpPromoConflict:503,innertubeResponseCacheMetrics:505,miniAppAdEvent:506,dataPlanUpsellEvent:507,producerProjectRenamed:508,producerMediaSelectionEvent:511,
embedsAutoplayStatusChanged:512,remoteConnectEvent:513,connectedSessionMisattributionEvent:514,producerProjectElementModified:515,adsSeenClientLogging:517,producerEvent:518,tvhtml5CleanStart:519,deviceAccountMetricsEvent:520,derpLogEvent:521,playablesPortalEvent:523,ipValidationStarted:524,ipValidationReceived:525,reelsSequenceMutationEvent:527,watchZoomStateChange:528,metadataEditorEvent:529,kidsPrismaDeeplinksEvent:530,creationOrchestrationEvent:531,coordinatedSamplingTriggered:532,dnaRecapScreenshotEvent:534,
mdxLocalNetworkPermissionRequestEvent:536,mdxLocalNetworkPermissionResponseEvent:537,sessionReplayEvent:538,sessionReplayStatusEvent:539};var Rv={},Sv=hu("ServiceWorkerLogsDatabase",{Nb:(Rv.SWHealthLog={Xb:1},Rv),shared:!0,upgrade:function(a,b){b(1)&&tt(mt(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function Tv(a){return Ht(Sv(),a)}
function Uv(a){var b,c;B(function(d){if(d.h==1)return d.yield(Tv(a),2);b=d.i;c=W()-2592E6;return d.yield(lt(b,["SWHealthLog"],{mode:"readwrite",ka:!0},function(e){return vt(e.objectStore("SWHealthLog"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return wt(f)})})}),0)})}
function Vv(a){var b;return B(function(c){if(c.h==1)return c.yield(Tv(a),2);b=c.i;return c.yield(b.clear("SWHealthLog"),0)})}
;var Wv={},Xv=0;function Yv(a){var b=b===void 0?{}:b;var c=new Image,d=""+Xv++;Wv[d]=c;c.onload=c.onerror=function(){delete Wv[d]};
b.yj&&(c.referrerPolicy="no-referrer");c.src=a}
;var Zv;function $v(){Zv||(Zv=new vs("yt.offline"));return Zv}
function aw(a){if(T("offline_error_handling")){var b=$v().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);$v().set("errors",b,2592E3,!0)}}
;function bw(){this.h=new Map;this.i=!1}
function cw(){if(!bw.instance){var a=G("yt.networkRequestMonitor.instance")||new bw;E("yt.networkRequestMonitor.instance",a);bw.instance=a}return bw.instance}
bw.prototype.requestComplete=function(a,b){b&&(this.i=!0);a=this.removeParams(a);this.h.get(a)||this.h.set(a,b)};
bw.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.h.get(a))?!1:a===!1&&this.i?!0:null};
bw.prototype.removeParams=function(a){return a.split("?")[0]};
bw.prototype.removeParams=bw.prototype.removeParams;bw.prototype.isEndpointCFR=bw.prototype.isEndpointCFR;bw.prototype.requestComplete=bw.prototype.requestComplete;bw.getInstance=cw;function dw(){pk.call(this);var a=this;this.j=!1;this.h=un();this.h.listen("networkstatus-online",function(){if(a.j&&T("offline_error_handling")){var b=$v().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new U(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;lq(d)}$v().set("errors",{},2592E3,!0)}}})}
v(dw,pk);function ew(){if(!dw.instance){var a=G("yt.networkStatusManager.instance")||new dw;E("yt.networkStatusManager.instance",a);dw.instance=a}return dw.instance}
p=dw.prototype;p.va=function(){return this.h.va()};
p.sb=function(a){this.h.h=a};
p.xf=function(){var a=window.navigator.onLine;return a===void 0?!0:a};
p.cf=function(){this.j=!0};
p.listen=function(a,b){return this.h.listen(a,b)};
p.Yc=function(a){a=sn(this.h,a);a.then(function(b){T("use_cfr_monitor")&&cw().requestComplete("generate_204",b)});
return a};
dw.prototype.sendNetworkCheckRequest=dw.prototype.Yc;dw.prototype.listen=dw.prototype.listen;dw.prototype.enableErrorFlushing=dw.prototype.cf;dw.prototype.getWindowStatus=dw.prototype.xf;dw.prototype.networkStatusHint=dw.prototype.sb;dw.prototype.isNetworkAvailable=dw.prototype.va;dw.getInstance=ew;function fw(a){a=a===void 0?{}:a;pk.call(this);var b=this;this.h=this.u=0;this.j=ew();var c=G("yt.networkStatusManager.instance.listen").bind(this.j);c&&(a.rateLimit?(this.rateLimit=a.rateLimit,c("networkstatus-online",function(){gw(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){gw(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){qk(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){qk(b,"publicytnetworkstatus-offline")})))}
v(fw,pk);fw.prototype.va=function(){var a=G("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.j)():!0};
fw.prototype.sb=function(a){var b=G("yt.networkStatusManager.instance.networkStatusHint").bind(this.j);b&&b(a)};
fw.prototype.Yc=function(a){var b=this,c;return B(function(d){c=G("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.j);return T("skip_network_check_if_cfr")&&cw().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.sb(((f=window.navigator)==null?void 0:f.onLine)||!0);e(b.va())})):c?d.return(c(a)):d.return(!0)})};
function gw(a,b){a.rateLimit?a.h?(vn.sa(a.u),a.u=vn.ra(function(){a.o!==b&&(qk(a,b),a.o=b,a.h=W())},a.rateLimit-(W()-a.h))):(qk(a,b),a.o=b,a.h=W()):qk(a,b)}
;var hw;function iw(){var a=rv.call;hw||(hw=new fw({kj:!0,Zi:!0}));a.call(rv,this,{ga:{Ue:Ov,Cb:Nv,Td:Kv,Lf:Lv,Bd:Mv,set:Iv},fa:hw,handleError:function(b,c,d){var e,f=d==null?void 0:(e=d.error)==null?void 0:e.code;if(f===400||f===415){var g;b=new U(b.message,c,d==null?void 0:(g=d.error)==null?void 0:g.code);mq(b,void 0,void 0,void 0,!0)}else lq(b)},
Fb:mq,sendFn:jw,now:W,He:aw,Ea:us(),Ad:"publicytnetworkstatus-online",xd:"publicytnetworkstatus-offline",Gc:!0,Ec:.1,Uc:Iq("potential_esf_error_limit",10),Y:T,dc:!(Lr()&&kw())});this.j=new Ml;T("networkless_immediately_drop_all_requests")&&Pv();eu("LogsDatabaseV2")}
v(iw,rv);function lw(){var a=G("yt.networklessRequestController.instance");a||(a=new iw,E("yt.networklessRequestController.instance",a),T("networkless_logging")&&Ut().then(function(b){a.W=b;tv(a);a.j.resolve();a.Gc&&Math.random()<=a.Ec&&a.W&&Uv(a.W);T("networkless_immediately_drop_sw_health_store")&&mw(a)}));
return a}
iw.prototype.writeThenSend=function(a,b){b||(b={});b=nw(a,b);Lr()||(this.h=!1);rv.prototype.writeThenSend.call(this,a,b)};
iw.prototype.sendThenWrite=function(a,b,c){b||(b={});b=nw(a,b);Lr()||(this.h=!1);rv.prototype.sendThenWrite.call(this,a,b,c)};
iw.prototype.sendAndWrite=function(a,b){b||(b={});b=nw(a,b);Lr()||(this.h=!1);rv.prototype.sendAndWrite.call(this,a,b)};
iw.prototype.awaitInitialization=function(){return this.j.promise};
function mw(a){var b;B(function(c){if(!a.W)throw b=Ws("clearSWHealthLogsDb"),b;return c.return(Vv(a.W).catch(function(d){a.handleError(d)}))})}
function jw(a,b,c,d){d=d===void 0?!1:d;b=T("web_fp_via_jspb")?Object.assign({},b):b;T("use_cfr_monitor")&&ow(a,b);if(T("use_request_time_ms_header"))b.headers&&xq(a)&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(W())));else{var e;if((e=b.postParams)==null?0:e.requestTimeMs)b.postParams.requestTimeMs=Math.round(W())}if(c&&Object.keys(b).length===0){var f=f===void 0?"":f;var g=g===void 0?!1:g;var h=h===void 0?!1:h;if(a)if(f)Pq(a,void 0,"POST",f,void 0);else if(S("USE_NET_AJAX_FOR_PING_TRANSPORT",
!1)||h)Pq(a,void 0,"GET","",void 0,void 0,g,h);else{b:{try{c:{var k=new db({url:a});if(k.h.dsh==="1")var l=null;else{var m=k.h.ae;if(m==="1"){var n=k.h.adurl;if(n)try{l={version:3,af:decodeURIComponent(n),Oe:bb(k.i,"act=1","ri=1",eb(k))};break c}catch(na){}}l=m==="2"?{version:4,af:bb(k.i,"dct=1","suid="+k.j,"ri=1"),Oe:bb(k.i,"act=1","ri=1","suid="+k.j)}:null}}if(l){var r=mc(a),t;if(!(t=!r||!r.endsWith("/aclk"))){var w=a.search(uc),z=tc(a,0,"ri",w);if(z<0)var x=null;else{var H=a.indexOf("&",z);if(H<
0||H>w)H=w;x=hc(a.slice(z+3,H!==-1?H:0))}t=x!=="1"}var F=!t;break b}}catch(na){}F=!1}if(F){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var V=!0;break b}}catch(na){}V=!1}c=V?!0:!1}else c=!1;c||Yv(a)}}else b.compress?b.postBody?(typeof b.postBody!=="string"&&(b.postBody=JSON.stringify(b.postBody)),gv(a,b.postBody,b,Tq,d)):gv(a,JSON.stringify(b.postParams),b,Sq,d):Tq(a,b)}
function nw(a,b){T("use_event_time_ms_header")&&xq(a)&&(b.headers||(b.headers={}),b.headers["X-Goog-Event-Time"]=JSON.stringify(Math.round(W())));return b}
function ow(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){cw().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){cw().requestComplete(a,!0);d(e,f)}}
function kw(){return lc(document.location.toString())!=="www.youtube-nocookie.com"}
;var pw=!1,qw=D.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:pw};E("ytNetworklessLoggingInitializationOptions",qw);function rw(){var a;B(function(b){if(b.h==1)return b.yield(Ut(),2);a=b.i;if(!a||!Lr()&&!T("nwl_init_require_datasync_id_killswitch")||!kw())return b.v(0);pw=!0;qw.isNwlInitialized=pw;return b.yield(lw().awaitInitialization(),0)})}
;function sw(a){var b=this;this.config_=null;a?this.config_=a:vu()&&(this.config_=wu());Or(function(){pv(b)},5E3)}
sw.prototype.isReady=function(){!this.config_&&vu()&&(this.config_=wu());return!!this.config_};
function qv(a,b,c,d){function e(n){n=n===void 0?!1:n;var r;if(d.retry&&h!="www.youtube-nocookie.com"&&(n||T("skip_ls_gel_retry")||g.headers["Content-Type"]!=="application/json"||(r=nv(b,c,l,k)),r)){var t=g.onSuccess,w=g.onFetchSuccess;g.onSuccess=function(H,F){ov(r);t(H,F)};
c.onFetchSuccess=function(H,F){ov(r);w(H,F)}}try{if(n&&d.retry&&!d.networklessOptions.bypassNetworkless)g.method="POST",d.networklessOptions.writeThenSend?lw().writeThenSend(m,g):lw().sendAndWrite(m,g);
else if(d.compress){var z=!d.networklessOptions.writeThenSend;if(g.postBody){var x=g.postBody;typeof x!=="string"&&(x=JSON.stringify(g.postBody));gv(m,x,g,Tq,z)}else gv(m,JSON.stringify(g.postParams),g,Sq,z)}else Sq(m,g)}catch(H){if(H.name==="InvalidAccessError")r&&(ov(r),r=0),mq(Error("An extension is blocking network request."));else throw H;}r&&Or(function(){pv(a)},5E3)}
!S("VISITOR_DATA")&&b!=="visitor_id"&&Math.random()<.01&&mq(new U("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new U("innertube xhrclient not ready",b,c,d);lq(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(n,r){if(d.onSuccess)d.onSuccess(r)},
onFetchSuccess:function(n){if(d.onSuccess)d.onSuccess(n)},
onProgress:function(n){if(d.onProgress)d.onProgress(n)},
onError:function(n,r){if(d.onError)d.onError(r)},
onFetchError:function(n){if(d.onError)d.onError(n)},
timeout:d.timeout,withCredentials:!0,compress:d.compress};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.Ef)&&(h=f);var k=a.config_.Ff||!1,l=yu(k,h,d);Object.assign(g.headers,l);g.headers.Authorization&&!h&&k&&(g.headers["x-origin"]=window.location.origin);var m=vq(""+h+("/youtubei/"+a.config_.innertubeApiVersion+"/"+b),{alt:"json"});(G("ytNetworklessLoggingInitializationOptions")?qw.isNwlInitialized:pw)?St().then(function(n){e(n)}):e(!1)}
;var tw=0,uw=yd?"webkit":xd?"moz":vd?"ms":ud?"o":"";E("ytDomDomGetNextId",G("ytDomDomGetNextId")||function(){return++tw});var vw={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function ww(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.scale=1;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in vw||(this[b]=a[b]);this.scale=a.scale;this.rotation=a.rotation;var c=a.target||a.srcElement;c&&c.nodeType==3&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;
if(d)try{d=d.nodeName?d:null}catch(e){d=null}else this.type=="mouseover"?d=a.fromElement:this.type=="mouseout"&&(d=a.toElement);this.relatedTarget=d;this.clientX=a.clientX!=void 0?a.clientX:a.pageX;this.clientY=a.clientY!=void 0?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||(this.type=="keypress"?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.h=a.pageX;this.i=a.pageY}}catch(e){}}
function xw(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.h=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.i=a.clientY+b}}
ww.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
ww.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
ww.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var Hi=D.ytEventsEventsListeners||{};E("ytEventsEventsListeners",Hi);var yw=D.ytEventsEventsCounter||{count:0};E("ytEventsEventsCounter",yw);
function zw(a,b,c,d){d=d===void 0?{}:d;a.addEventListener&&(b!="mouseenter"||"onmouseenter"in document?b!="mouseleave"||"onmouseenter"in document?b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return Gi(function(e){var f=typeof e[4]==="boolean"&&e[4]==!!d,g=Pa(e[4])&&Pa(d)&&Li(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
function Aw(a,b,c,d){d=d===void 0?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=zw(a,b,c,d);if(e)return e;e=++yw.count+"";var f=!(b!="mouseenter"&&b!="mouseleave"||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new ww(h);if(!Wi(h.relatedTarget,function(k){return k==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new ww(h);
h.currentTarget=a;return c.call(a,h)};
g=kq(g);a.addEventListener?(b=="mouseenter"&&f?b="mouseover":b=="mouseleave"&&f?b="mouseout":b=="mousewheel"&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),Bw()||typeof d==="boolean"?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);Hi[e]=[a,b,c,g,d];return e}
function Cw(a){a&&(typeof a=="string"&&(a=[a]),Tb(a,function(b){if(b in Hi){var c=Hi[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?Bw()||typeof c==="boolean"?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete Hi[b]}}))}
var Bw=Fk(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});function Dw(a){this.G=a;this.h=null;this.o=0;this.B=null;this.u=0;this.i=[];for(a=0;a<4;a++)this.i.push(0);this.j=0;this.V=Aw(window,"mousemove",Xa(this.aa,this));a=Xa(this.K,this);typeof a==="function"&&(a=kq(a));this.ba=window.setInterval(a,25)}
ab(Dw,J);Dw.prototype.aa=function(a){a.h===void 0&&xw(a);var b=a.h;a.i===void 0&&xw(a);this.h=new Ci(b,a.i)};
Dw.prototype.K=function(){if(this.h){var a=W();if(this.o!=0){var b=this.B,c=this.h,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.o);this.i[this.j]=Math.abs((d-this.u)/this.u)>.5?1:0;for(c=b=0;c<4;c++)b+=this.i[c]||0;b>=3&&this.G();this.u=d}this.o=a;this.B=this.h;this.j=(this.j+1)%4}};
Dw.prototype.X=function(){window.clearInterval(this.ba);Cw(this.V)};var Ew={};function Fw(a){var b=a===void 0?{}:a;a=b.ag===void 0?!1:b.ag;b=b.df===void 0?!0:b.df;if(G("_lact",window)==null){var c=parseInt(S("LACT"),10);c=isFinite(c)?Date.now()-Math.max(c,0):-1;E("_lact",c,window);E("_fact",c,window);c==-1&&Gw();Hw(a,b);new Dw(function(){Iw("mouse",100)})}}
function Hw(a,b){var c=window;a=a===void 0?!1:a;b=b===void 0?!0:b;Aw(c.document,"keydown",Gw);Aw(c.document,"keyup",Gw);Aw(c.document,"mousedown",Gw);Aw(c.document,"mouseup",Gw);a?Aw(c,"touchmove",function(){Iw("touchmove",200)},{passive:!0}):(Aw(c,"resize",function(){Iw("resize",200)}),b&&Aw(c,"scroll",function(){Iw("scroll",200)}));
Aw(c.document,"touchstart",Gw,{passive:!0});Aw(c.document,"touchend",Gw,{passive:!0})}
function Iw(a,b){Ew[a]||(Ew[a]=!0,vn.ra(function(){Gw();Ew[a]=!1},b))}
function Gw(){G("_lact",window)==null&&Fw();var a=Date.now();E("_lact",a,window);G("_fact",window)==-1&&E("_fact",a,window);(a=G("ytglobal.ytUtilActivityCallback_"))&&a()}
function Jw(){var a=G("_lact",window);return a==null?-1:Math.max(Date.now()-a,0)}
;var Kw=D.ytPubsubPubsubInstance||new O,Lw=D.ytPubsubPubsubSubscribedKeys||{},Mw=D.ytPubsubPubsubTopicToKeys||{},Nw=D.ytPubsubPubsubIsSynchronous||{};function Ow(a,b){var c=Pw();if(c&&b){var d=c.subscribe(a,function(){function e(){Lw[d]&&b.apply&&typeof b.apply=="function"&&b.apply(window,f)}
var f=arguments;try{Nw[a]?e():Eq(e,0)}catch(g){lq(g)}},void 0);
Lw[d]=!0;Mw[a]||(Mw[a]=[]);Mw[a].push(d);return d}return 0}
function Qw(a){var b=Pw();b&&(typeof a==="number"?a=[a]:typeof a==="string"&&(a=[parseInt(a,10)]),Tb(a,function(c){b.unsubscribeByKey(c);delete Lw[c]}))}
function Rw(a,b){var c=Pw();c&&c.publish.apply(c,arguments)}
function Sw(a){var b=Pw();if(b)if(b.clear(a),a)Tw(a);else for(var c in Mw)Tw(c)}
function Pw(){return D.ytPubsubPubsubInstance}
function Tw(a){Mw[a]&&(a=Mw[a],Tb(a,function(b){Lw[b]&&delete Lw[b]}),a.length=0)}
O.prototype.subscribe=O.prototype.subscribe;O.prototype.unsubscribeByKey=O.prototype.xc;O.prototype.publish=O.prototype.yb;O.prototype.clear=O.prototype.clear;E("ytPubsubPubsubInstance",Kw);E("ytPubsubPubsubTopicToKeys",Mw);E("ytPubsubPubsubIsSynchronous",Nw);E("ytPubsubPubsubSubscribedKeys",Lw);var Uw=Symbol("injectionDeps");function Vw(a){this.name=a}
Vw.prototype.toString=function(){return"InjectionToken("+this.name+")"};
function Ww(a){this.key=a}
function Xw(){this.i=new Map;this.j=new Map;this.h=new Map}
function Yw(a,b){a.i.set(b.oc,b);var c=a.j.get(b.oc);if(c)try{c.Xc(a.resolve(b.oc))}catch(d){c.uj(d)}}
Xw.prototype.resolve=function(a){return a instanceof Ww?Zw(this,a.key,[],!0):Zw(this,a,[])};
function Zw(a,b,c,d){d=d===void 0?!1:d;if(c.indexOf(b)>-1)throw Error("Deps cycle for: "+b);if(a.h.has(b))return a.h.get(b);if(!a.i.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.i.get(b);c.push(b);if(d.ze!==void 0)var e=d.ze;else if(d.Hg)e=d[Uw]?$w(a,d[Uw],c):[],e=d.Hg.apply(d,A(e));else if(d.Gd){e=d.Gd;var f=e[Uw]?$w(a,e[Uw],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(A(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.Cj||a.h.set(b,e);return e}
function $w(a,b,c){return b?b.map(function(d){return d instanceof Ww?Zw(a,d.key,c,!0):Zw(a,d,c)}):[]}
;var ax;function bx(){ax||(ax=new Xw);return ax}
;var cx=window;function dx(){var a,b;return"h5vcc"in cx&&((a=cx.h5vcc.traceEvent)==null?0:a.traceBegin)&&((b=cx.h5vcc.traceEvent)==null?0:b.traceEnd)?1:"performance"in cx&&cx.performance.mark&&cx.performance.measure?2:0}
function ex(a){var b=dx();switch(b){case 1:cx.h5vcc.traceEvent.traceBegin("YTLR",a);break;case 2:cx.performance.mark(a+"-start");break;case 0:break;default:Eb(b,"unknown trace type")}}
function fx(a){var b=dx();switch(b){case 1:cx.h5vcc.traceEvent.traceEnd("YTLR",a);break;case 2:b=a+"-start";var c=a+"-end";cx.performance.mark(c);cx.performance.measure(a,b,c);break;case 0:break;default:Eb(b,"unknown trace type")}}
;var gx=T("web_enable_lifecycle_monitoring")&&dx()!==0,hx=T("web_enable_lifecycle_monitoring");function ix(a){var b,c;(c=(b=window).onerror)==null||c.call(b,a.message,"",0,0,a)}
;function jx(a){var b=this;var c=c===void 0?0:c;var d=d===void 0?us():d;this.j=c;this.scheduler=d;this.i=new Ml;this.h=a;for(a={pb:0};a.pb<this.h.length;a={Tc:void 0,pb:a.pb},a.pb++)a.Tc=this.h[a.pb],c=function(e){return function(){e.Tc.qd();b.h[e.pb].Vc=!0;b.h.every(function(f){return f.Vc===!0})&&b.i.resolve()}}(a),d=this.getPriority(a.Tc),d=this.scheduler.Xa(c,d),this.h[a.pb]=Object.assign({},a.Tc,{qd:c,
jobId:d})}
function kx(a){var b=Array.from(a.h.keys()).sort(function(d,e){return a.getPriority(a.h[e])-a.getPriority(a.h[d])});
b=y(b);for(var c=b.next();!c.done;c=b.next())c=a.h[c.value],c.jobId===void 0||c.Vc||(a.scheduler.sa(c.jobId),a.scheduler.Xa(c.qd,10))}
jx.prototype.cancel=function(){for(var a=y(this.h),b=a.next();!b.done;b=a.next())b=b.value,b.jobId===void 0||b.Vc||this.scheduler.sa(b.jobId),b.Vc=!0;this.i.resolve()};
jx.prototype.getPriority=function(a){var b;return(b=a.priority)!=null?b:this.j};function lx(a){this.state=a;this.plugins=[];this.o=void 0;this.B={};gx&&ex(this.state)}
p=lx.prototype;p.install=function(a){this.plugins.push(a);return this};
p.uninstall=function(){var a=this;C.apply(0,arguments).forEach(function(b){b=a.plugins.indexOf(b);b>-1&&a.plugins.splice(b,1)})};
p.transition=function(a,b){var c=this;gx&&fx(this.state);var d=this.transitions.find(function(f){return Array.isArray(f.from)?f.from.find(function(g){return g===c.state&&f.to===a}):f.from===c.state&&f.to===a});
if(d){this.j&&(kx(this.j),this.j=void 0);mx(this,a,b);this.state=a;gx&&ex(this.state);d=d.action.bind(this);var e=this.plugins.filter(function(f){return f[a]}).map(function(f){return f[a]});
d(nx(this,e),b)}else throw Error("no transition specified from "+this.state+" to "+a);};
function nx(a,b){var c=b.filter(function(e){return ox(a,e)===10}),d=b.filter(function(e){return ox(a,e)!==10});
return a.B.Bj?function(){var e=C.apply(0,arguments);return B(function(f){if(f.h==1)return f.yield(a.hg.apply(a,[c].concat(A(e))),2);a.re.apply(a,[d].concat(A(e)));f.h=0})}:function(){var e=C.apply(0,arguments);
a.ig.apply(a,[c].concat(A(e)));a.re.apply(a,[d].concat(A(e)))}}
p.ig=function(a){for(var b=C.apply(1,arguments),c=us(),d=y(a),e=d.next(),f={};!e.done;f={fc:void 0},e=d.next())f.fc=e.value,c.Vb(function(g){return function(){px(g.fc.name);qx(function(){return g.fc.callback.apply(g.fc,A(b))});
rx(g.fc.name)}}(f))};
p.hg=function(a){var b=C.apply(1,arguments),c,d,e,f,g;return B(function(h){h.h==1&&(c=us(),d=y(a),e=d.next(),f={});if(h.h!=3){if(e.done)return h.v(0);f.eb=e.value;f.yc=void 0;g=function(k){return function(){px(k.eb.name);var l=qx(function(){return k.eb.callback.apply(k.eb,A(b))});
Ee(l)?k.yc=T("web_lifecycle_error_handling_killswitch")?l.then(function(){rx(k.eb.name)}):l.then(function(){rx(k.eb.name)},function(m){ix(m);
rx(k.eb.name)}):rx(k.eb.name)}}(f);
c.Vb(g);return f.yc?h.yield(f.yc,3):h.v(3)}f={eb:void 0,yc:void 0};e=d.next();return h.v(2)})};
p.re=function(a){var b=C.apply(1,arguments),c=this,d=a.map(function(e){return{qd:function(){px(e.name);qx(function(){return e.callback.apply(e,A(b))});
rx(e.name)},
priority:ox(c,e)}});
d.length&&(this.j=new jx(d))};
function ox(a,b){var c,d;return(d=(c=a.o)!=null?c:b.priority)!=null?d:0}
function px(a){gx&&a&&ex(a)}
function rx(a){gx&&a&&fx(a)}
function mx(a,b,c){hx&&console.groupCollapsed&&console.groupEnd&&(console.groupCollapsed("["+a.constructor.name+"] '"+a.state+"' to '"+b+"'"),console.log("with message: ",c),console.groupEnd())}
da.Object.defineProperties(lx.prototype,{currentState:{configurable:!0,enumerable:!0,get:function(){return this.state}}});
function qx(a){if(T("web_lifecycle_error_handling_killswitch"))return a();try{return a()}catch(b){ix(b)}}
;function sx(a){lx.call(this,a===void 0?"none":a);this.h=null;this.o=10;this.transitions=[{from:"none",to:"application_navigating",action:this.i},{from:"application_navigating",to:"none",action:this.u},{from:"application_navigating",to:"application_navigating",action:function(){}},
{from:"none",to:"none",action:function(){}}]}
var tx;v(sx,lx);sx.prototype.i=function(a,b){var c=this;this.h=Or(function(){c.currentState==="application_navigating"&&c.transition("none")},5E3);
a(b==null?void 0:b.event)};
sx.prototype.u=function(a,b){this.h&&(vn.sa(this.h),this.h=null);a(b==null?void 0:b.event)};
function ux(){tx||(tx=new sx);return tx}
;var vx=[];E("yt.logging.transport.getScrapedGelPayloads",function(){return vx});function wx(){this.store={};this.h={}}
wx.prototype.storePayload=function(a,b){a=xx(a);this.store[a]?this.store[a].push(b):(this.h={},this.store[a]=[b]);T("more_accurate_gel_parser")&&(b=new CustomEvent("TRANSPORTING_NEW_EVENT"),window.dispatchEvent(b));return a};
wx.prototype.smartExtractMatchingEntries=function(a){if(!a.keys.length)return[];for(var b=yx(this,a.keys.splice(0,1)[0]),c=[],d=0;d<b.length;d++)this.store[b[d]]&&a.sizeLimit&&(this.store[b[d]].length<=a.sizeLimit?(c.push.apply(c,A(this.store[b[d]])),delete this.store[b[d]]):c.push.apply(c,A(this.store[b[d]].splice(0,a.sizeLimit))));(a==null?0:a.sizeLimit)&&c.length<(a==null?void 0:a.sizeLimit)&&(a.sizeLimit-=c.length,c.push.apply(c,A(this.smartExtractMatchingEntries(a))));return c};
wx.prototype.extractMatchingEntries=function(a){a=yx(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,A(this.store[a[c]])),delete this.store[a[c]]);return b};
wx.prototype.getSequenceCount=function(a){a=yx(this,a);for(var b=0,c=0;c<a.length;c++){var d=void 0;b+=((d=this.store[a[c]])==null?void 0:d.length)||0}return b};
function yx(a,b){var c=xx(b);if(a.h[c])return a.h[c];var d=Object.keys(a.store)||[];if(d.length<=1&&xx(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(zx(b.auth,g[0])){var h=b.isJspb;zx(h===void 0?"undefined":h?"true":"false",g[1])&&zx(b.cttAuthInfo,g[2])&&(h=b.tier,h=h===void 0?"undefined":JSON.stringify(h),zx(h,g[3])&&e.push(d[f]))}}return a.h[c]=e}
function zx(a,b){return a===void 0||a==="undefined"?!0:a===b}
wx.prototype.getSequenceCount=wx.prototype.getSequenceCount;wx.prototype.extractMatchingEntries=wx.prototype.extractMatchingEntries;wx.prototype.smartExtractMatchingEntries=wx.prototype.smartExtractMatchingEntries;wx.prototype.storePayload=wx.prototype.storePayload;function xx(a){return[a.auth===void 0?"undefined":a.auth,a.isJspb===void 0?"undefined":a.isJspb,a.cttAuthInfo===void 0?"undefined":a.cttAuthInfo,a.tier===void 0?"undefined":a.tier].join("/")}
;function Ax(a,b){if(a)return a[b.name]}
;var Bx=Iq("initial_gel_batch_timeout",2E3),Cx=Iq("gel_queue_timeout_max_ms",6E4),Dx=Iq("gel_min_batch_size",5),Ex=void 0;function Fx(){this.o=this.h=this.i=0;this.j=!1}
var Gx=new Fx,Hx=new Fx,Ix=new Fx,Jx=new Fx,Kx,Lx=!0,Mx=D.ytLoggingTransportTokensToCttTargetIds_||{};E("ytLoggingTransportTokensToCttTargetIds_",Mx);var Nx={};function Ox(){var a=G("yt.logging.ims");a||(a=new wx,E("yt.logging.ims",a));return a}
function Px(a,b){if(a.endpoint==="log_event"){Qx(a);var c=Rx(a),d=Sx(a.payload)||"";a:{if(T("enable_web_tiered_gel")){var e=Qv[d||""];var f,g,h,k=bx().resolve(new Ww(qu))==null?void 0:(f=ru())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.eventLoggingConfig)==null?void 0:h.payloadPolicies;if(k)for(f=0;f<k.length;f++)if(k[f].payloadNumber===e){e=k[f];break a}}e=void 0}k=200;if(e){if(e.enabled===!1&&!T("web_payload_policy_disabled_killswitch"))return;k=Tx(e.tier);if(k===400){Ux(a,b);return}}Nx[c]=
!0;c={cttAuthInfo:c,isJspb:!1,tier:k};Ox().storePayload(c,a.payload);Vx(b,c,d==="gelDebuggingEvent")}}
function Vx(a,b,c){function d(){Wx({writeThenSend:!0},void 0,e,b.tier)}
var e=!1;e=e===void 0?!1:e;c=c===void 0?!1:c;a&&(Ex=new a);a=Iq("tvhtml5_logging_max_batch_ads_fork")||Iq("tvhtml5_logging_max_batch")||Iq("web_logging_max_batch")||100;var f=W(),g=Xx(e,b.tier),h=g.o;c&&(g.j=!0);c=0;b&&(c=Ox().getSequenceCount(b));c>=1E3?d():c>=a?Kx||(Kx=Yx(function(){d();Kx=void 0},0)):f-h>=10&&(Zx(e,b.tier),g.o=f)}
function Ux(a,b){if(a.endpoint==="log_event"){T("more_accurate_gel_parser")&&Ox().storePayload({isJspb:!1},a.payload);Qx(a);var c=Rx(a),d=new Map;d.set(c,[a.payload]);var e=Sx(a.payload)||"";b&&(Ex=new b);return new Gk(function(f,g){Ex&&Ex.isReady()?$x(d,Ex,f,g,{bypassNetworkless:!0},!0,e==="gelDebuggingEvent"):f()})}}
function Rx(a){var b="";if(a.dangerousLogToVisitorSession)b="visitorOnlyApprovedKey";else if(a.cttAuthInfo){b=a.cttAuthInfo;var c={};b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId);Mx[a.cttAuthInfo.token]=c;b=a.cttAuthInfo.token}return b}
function Wx(a,b,c,d){a=a===void 0?{}:a;c=c===void 0?!1:c;new Gk(function(e,f){var g=Xx(c,d),h=g.j;g.j=!1;ay(g.i);ay(g.h);g.h=0;Ex&&Ex.isReady()?d===void 0&&T("enable_web_tiered_gel")?by(e,f,a,b,c,300,h):by(e,f,a,b,c,d,h):(Zx(c,d),e())})}
function by(a,b,c,d,e,f,g){var h=Ex;c=c===void 0?{}:c;e=e===void 0?!1:e;f=f===void 0?200:f;g=g===void 0?!1:g;var k=new Map,l={isJspb:e,cttAuthInfo:d,tier:f};e={isJspb:e,cttAuthInfo:d};if(d!==void 0)f=T("enable_web_tiered_gel")?Ox().smartExtractMatchingEntries({keys:[l,e],sizeLimit:1E3}):Ox().extractMatchingEntries(e),k.set(d,f);else for(d=y(Object.keys(Nx)),l=d.next();!l.done;l=d.next())l=l.value,e=T("enable_web_tiered_gel")?Ox().smartExtractMatchingEntries({keys:[{isJspb:!1,cttAuthInfo:l,tier:f},
{isJspb:!1,cttAuthInfo:l}],sizeLimit:1E3}):Ox().extractMatchingEntries({isJspb:!1,cttAuthInfo:l}),e.length>0&&k.set(l,e),(T("web_fp_via_jspb_and_json")&&c.writeThenSend||!T("web_fp_via_jspb_and_json"))&&delete Nx[l];$x(k,h,a,b,c,!1,g)}
function Zx(a,b){function c(){Wx({writeThenSend:!0},void 0,a,b)}
a=a===void 0?!1:a;b=b===void 0?200:b;var d=Xx(a,b),e=d===Jx||d===Ix?5E3:Cx;T("web_gel_timeout_cap")&&!d.h&&(e=Yx(function(){c()},e),d.h=e);
ay(d.i);e=S("LOGGING_BATCH_TIMEOUT",Iq("web_gel_debounce_ms",1E4));T("shorten_initial_gel_batch_timeout")&&Lx&&(e=Bx);e=Yx(function(){Iq("gel_min_batch_size")>0?Ox().getSequenceCount({cttAuthInfo:void 0,isJspb:a,tier:b})>=Dx&&c():c()},e);
d.i=e}
function $x(a,b,c,d,e,f,g){e=e===void 0?{}:e;var h=Math.round(W()),k=a.size,l=(g===void 0?0:g)&&T("vss_through_gel_video_stats")?"video_stats":"log_event";a=y(a);var m=a.next();for(g={};!m.done;g={vd:void 0,batchRequest:void 0,dangerousLogToVisitorSession:void 0,zd:void 0,yd:void 0},m=a.next()){var n=y(m.value);m=n.next().value;n=n.next().value;g.batchRequest=Ni({context:xu(b.config_||wu())});if(!Oa(n)&&!T("throw_err_when_logevent_malformed_killswitch")){d();break}g.batchRequest.events=n;(n=Mx[m])&&
cy(g.batchRequest,m,n);delete Mx[m];g.dangerousLogToVisitorSession=m==="visitorOnlyApprovedKey";dy(g.batchRequest,h,g.dangerousLogToVisitorSession);T("always_send_and_write")&&(e.writeThenSend=!1);g.zd=function(r){T("start_client_gcf")&&vn.ra(function(){return B(function(t){return t.yield(ey(r),0)})});
k--;k||c()};
g.vd=0;g.yd=function(r){return function(){r.vd++;if(e.bypassNetworkless&&r.vd===1)try{qv(b,l,r.batchRequest,fy({writeThenSend:!0},r.dangerousLogToVisitorSession,r.zd,r.yd,f)),Lx=!1}catch(t){lq(t),d()}k--;k||c()}}(g);
try{qv(b,l,g.batchRequest,fy(e,g.dangerousLogToVisitorSession,g.zd,g.yd,f)),Lx=!1}catch(r){lq(r),d()}}}
function fy(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,networklessOptions:a,dangerousLogToVisitorSession:b,Ui:!!e,headers:{},postBodyFormat:"",postBody:"",compress:T("compress_gel")||T("compress_gel_lr")};gy()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(W())));return a}
function dy(a,b,c){gy()||(a.requestTimeMs=String(b));T("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=S("EVENT_ID"))&&((c=S("BATCH_CLIENT_COUNTER")||0)||(c=Math.floor(Math.random()*65535/2)),c++,c>65535&&(c=1),gq("BATCH_CLIENT_COUNTER",c),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function cy(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function Qx(a){var b=Kq("il_payload_scraping")==="enable_il_payload_scraping";if(!G("yt.logging.transport.enableScrapingForTest"))if(b)vx=[],E("yt.logging.transport.enableScrapingForTest",!0),E("yt.logging.transport.scrapedPayloadsForTesting",vx),E("yt.logging.transport.payloadToScrape","visualElementShown visualElementHidden visualElementAttached screenCreated visualElementGestured visualElementStateChanged".split(" ")),E("yt.logging.transport.getScrapedPayloadFromClientEventsFunction"),E("yt.logging.transport.scrapeClientEvent",
!0);else return;b=G("yt.logging.transport.scrapedPayloadsForTesting");var c=G("yt.logging.transport.payloadToScrape"),d=G("yt.logging.transport.scrapeClientEvent");if(c&&c.length>=1)for(var e=0;e<c.length;e++)if(a&&a.payload[c[e]])if(d)b.push(a.payload);else{var f=void 0;b.push(((f=a)==null?void 0:f.payload)[c[e]])}E("yt.logging.transport.scrapedPayloadsForTesting",b)}
function gy(){return T("use_request_time_ms_header")||T("lr_use_request_time_ms_header")}
function Yx(a,b){return T("transport_use_scheduler")===!1?Eq(a,b):T("logging_avoid_blocking_during_navigation")||T("lr_logging_avoid_blocking_during_navigation")?Or(function(){if(ux().currentState==="none")a();else{var c={};ux().install((c.none={callback:a},c))}},b):Or(a,b)}
function ay(a){T("transport_use_scheduler")?vn.sa(a):window.clearTimeout(a)}
function ey(a){var b,c,d,e,f,g,h,k,l,m;return B(function(n){return n.h==1?(d=(b=a)==null?void 0:(c=b.responseContext)==null?void 0:c.globalConfigGroup,e=Ax(d,Ip),g=(f=d)==null?void 0:f.hotHashData,h=Ax(d,Hp),l=(k=d)==null?void 0:k.coldHashData,(m=bx().resolve(new Ww(qu)))?g?e?n.yield(su(m,g,e),2):n.yield(su(m,g),2):n.v(2):n.return()):l?h?n.yield(tu(m,l,h),0):n.yield(tu(m,l),0):n.v(0)})}
function Xx(a,b){b=b===void 0?200:b;return a?b===300?Jx:Hx:b===300?Ix:Gx}
function Sx(a){a=Object.keys(a);a=y(a);for(var b=a.next();!b.done;b=a.next())if(b=b.value,Qv[b])return b}
function Tx(a){switch(a){case "DELAYED_EVENT_TIER_UNSPECIFIED":return 0;case "DELAYED_EVENT_TIER_DEFAULT":return 100;case "DELAYED_EVENT_TIER_DISPATCH_TO_EMPTY":return 200;case "DELAYED_EVENT_TIER_FAST":return 300;case "DELAYED_EVENT_TIER_IMMEDIATE":return 400;default:return 200}}
;var hy=D.ytLoggingGelSequenceIdObj_||{};E("ytLoggingGelSequenceIdObj_",hy);
function iy(a,b,c,d){d=d===void 0?{}:d;var e={},f=Math.round(d.timestamp||W());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;a=Jw();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};d.sequenceGroup&&!T("web_gel_sequence_info_killswitch")&&(a=e.context,b=d.sequenceGroup,hy[b]=b in hy?hy[b]+1:0,a.sequence={index:hy[b],groupKey:b},d.endOfSequence&&delete hy[d.sequenceGroup]);T("web_tag_automated_log_events")&&(e.context.automatedLogEventSource=d.automatedLogEventSource);(d.sendIsolatedPayload?
Ux:Px)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,dangerousLogToVisitorSession:d.dangerousLogToVisitorSession},c)}
;function Es(a,b,c){c=c===void 0?{}:c;var d=sw;S("ytLoggingEventsDefaultDisabled",!1)&&sw===sw&&(d=null);iy(a,b,d,c)}
;var jy=new Set,ky=0,ly=0,my=0,ny=[],oy=[],py=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function qy(){Tb(S("ERRORS")||[],function(a){ry.apply(null,a)});
gq("ERRORS",[])}
function Ds(a){ry(a)}
function sy(a){ry(a,"WARNING")}
function ty(a){a instanceof Error?ry(a):(a=Pa(a)?JSON.stringify(a):String(a),a=new U(a),a.name="RejectedPromiseError",sy(a))}
function uy(a,b,c,d,e,f){b=b===void 0?"Unknown file":b;c=c===void 0?0:c;var g=!1,h=hq("log_window_onerror_fraction");if(h&&Math.random()<h)g=!0;else{h=document.getElementsByTagName("script");for(var k=0,l=h.length;k<l;k++)if(h[k].src.indexOf("/debug-")>0){g=!0;break}}if(g){g=!1;e?g=!0:(typeof a==="string"?h=a:ErrorEvent&&a instanceof ErrorEvent?(g=!0,h=a.message,b=a.filename,c=a.lineno,d=a.colno):(h="Unknown error",b="Unknown file",c=0),e=new U(h),e.name="UnhandledWindowError",e.message=h,e.fileName=
b,e.lineNumber=c,isNaN(d)?delete e.columnNumber:e.columnNumber=d);if(!T("wiz_enable_component_stack_propagation_killswitch")){a=e;var m;if((m=f)==null||!m.componentStack)if(m=a.We)f||(f={}),f.componentStack=m}f&&vy(e,f);g?ry(e):sy(e)}}
function ry(a,b,c,d,e,f,g,h){f=f===void 0?{}:f;f.name=c||S("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||S("INNERTUBE_CONTEXT_CLIENT_VERSION");c=f;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;b=b===void 0?"ERROR":b;g=g===void 0?!1:g;if(a&&(a.hasOwnProperty("level")&&a.level&&(b=a.level),T("console_log_js_exceptions")&&(d=[],d.push("Name: "+a.name),d.push("Message: "+a.message),a.hasOwnProperty("params")&&d.push("Error Params: "+JSON.stringify(a.params)),a.hasOwnProperty("args")&&d.push("Error args: "+
JSON.stringify(a.args)),d.push("File name: "+a.fileName),d.push("Stacktrace: "+a.stack),d=d.join("\n"),window.console.log(d,a)),!(ky>=5))){d=[];e=y(oy);for(f=e.next();!f.done;f=e.next()){f=f.value;try{f()&&d.push(f())}catch(x){}}d=[].concat(A(ny),A(d));var k=cc(a);e=k.message||"Unknown Error";f=k.name||"UnknownError";var l=k.stack||a.i||"Not available";if(l.startsWith(f+": "+e)){var m=l.split("\n");m.shift();l=m.join("\n")}m=k.lineNumber||"Not available";k=k.fileName||"Not available";var n=0;if(a.hasOwnProperty("args")&&
a.args&&a.args.length)for(var r=0;r<a.args.length&&!(n=mr(a.args[r],"params."+r,c,n),n>=500);r++);else if(a.hasOwnProperty("params")&&a.params){var t=a.params;if(typeof a.params==="object")for(r in t){if(t[r]){var w="params."+r,z=or(t[r]);c[w]=z;n+=w.length+z.length;if(n>500)break}}else c.params=or(t)}if(d.length)for(r=0;r<d.length&&!(n=mr(d[r],"params.context."+r,c,n),n>=500);r++);navigator.vendor&&!c.hasOwnProperty("vendor")&&(c["device.vendor"]=navigator.vendor);r={message:e,name:f,lineNumber:m,
fileName:k,stack:l,params:c,sampleWeight:1};c=Number(a.columnNumber);isNaN(c)||(r.lineNumber=r.lineNumber+":"+c);if(a.level==="IGNORED")a=0;else a:{a=hr();c=y(a.fb);for(d=c.next();!d.done;d=c.next())if(d=d.value,r.message&&r.message.match(d.oj)){a=d.weight;break a}a=y(a.Za);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.callback(r)){a=c.weight;break a}a=1}r.sampleWeight=a;a=y(br);for(c=a.next();!c.done;c=a.next())if(c=c.value,c.Sc[r.name])for(e=y(c.Sc[r.name]),d=e.next();!d.done;d=e.next())if(f=
d.value,d=r.message.match(f.regexp)){r.params["params.error.original"]=d[0];e=f.groups;f={};for(m=0;m<e.length;m++)f[e[m]]=d[m+1],r.params["params.error."+e[m]]=d[m+1];r.message=c.ud(f);break}r.params||(r.params={});a=hr();r.params["params.errorServiceSignature"]="msg="+a.fb.length+"&cb="+a.Za.length;r.params["params.serviceWorker"]="false";D.document&&D.document.querySelectorAll&&(r.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));(new Qi(Ri,"sample")).constructor!==
Qi&&(r.params["params.fconst"]="true");window.yterr&&typeof window.yterr==="function"&&window.yterr(r);if(r.sampleWeight!==0&&!jy.has(r.message)){if(g)wy(b===void 0?"ERROR":b,r);else{b=b===void 0?"ERROR":b;b==="ERROR"?(jr.yb("handleError",r),T("record_app_crashed_web")&&my===0&&r.sampleWeight===1&&(my++,g={appCrashType:"APP_CRASH_TYPE_BREAKPAD"},T("report_client_error_with_app_crash_ks")||(g.systemHealth={crashData:{clientError:{logMessage:{message:r.message}}}}),Es("appCrashed",g)),ly++):b==="WARNING"&&
jr.yb("handleWarning",r);if(T("kevlar_gel_error_routing")){g=b;h=h===void 0?{}:h;b:{a=y(py);for(c=a.next();!c.done;c=a.next())if(Ks(c.value.toLowerCase())){a=!0;break b}a=!1}if(a)h=void 0;else{c={stackTrace:r.stack};r.fileName&&(c.filename=r.fileName);a=r.lineNumber&&r.lineNumber.split?r.lineNumber.split(":"):[];a.length!==0&&(a.length!==1||isNaN(Number(a[0]))?a.length!==2||isNaN(Number(a[0]))||isNaN(Number(a[1]))||(c.lineNumber=Number(a[0]),c.columnNumber=Number(a[1])):c.lineNumber=Number(a[0]));
a={level:"ERROR_LEVEL_UNKNOWN",message:r.message,errorClassName:r.name,sampleWeight:r.sampleWeight};g==="ERROR"?a.level="ERROR_LEVEL_ERROR":g==="WARNING"&&(a.level="ERROR_LEVEL_WARNNING");c={isObfuscated:!0,browserStackInfo:c};h.pageUrl=window.location.href;h.kvPairs=[];S("FEXP_EXPERIMENTS")&&(h.experimentIds=S("FEXP_EXPERIMENTS"));d=S("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!hq("web_disable_gel_stp_ecatcher_killswitch")&&d)for(e=y(Object.keys(d)),f=e.next();!f.done;f=e.next())f=f.value,h.kvPairs.push({key:f,
value:String(d[f])});if(d=r.params)for(e=y(Object.keys(d)),f=e.next();!f.done;f=e.next())f=f.value,h.kvPairs.push({key:"client."+f,value:String(d[f])});d=S("SERVER_NAME");e=S("SERVER_VERSION");d&&e&&(h.kvPairs.push({key:"server.name",value:d}),h.kvPairs.push({key:"server.version",value:e}));h={errorMetadata:h,stackTrace:c,logMessage:a}}h&&(Es("clientError",h),(g==="ERROR"||T("errors_flush_gel_always_killswitch"))&&Wx(void 0,void 0,!1))}T("suppress_error_204_logging")||wy(b,r)}try{jy.add(r.message)}catch(x){}ky++}}}
function wy(a,b){var c=b.params||{};a={urlParams:{a:"logerror",t:"jserror",type:b.name,msg:b.message.substr(0,250),line:b.lineNumber,level:a,"client.name":c.name},postParams:{url:S("PAGE_NAME",window.location.href),file:b.fileName},method:"POST"};c.version&&(a["client.version"]=c.version);if(a.postParams){b.stack&&(a.postParams.stack=b.stack);b=y(Object.keys(c));for(var d=b.next();!d.done;d=b.next())d=d.value,a.postParams["client."+d]=c[d];if(c=S("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS"))for(b=y(Object.keys(c)),
d=b.next();!d.done;d=b.next())d=d.value,a.postParams[d]=c[d];(c=S("LAVA_VERSION"))&&(a.postParams["lava.version"]=c);c=S("SERVER_NAME");b=S("SERVER_VERSION");c&&b&&(a.postParams["server.name"]=c,a.postParams["server.version"]=b)}Tq(S("ECATCHER_REPORT_HOST","")+"/error_204",a)}
function vy(a){var b=C.apply(1,arguments);a.args||(a.args=[]);Array.isArray(a.args)&&a.args.push.apply(a.args,A(b))}
;function xy(){this.register=new Map}
function yy(a){a=y(a.register.values());for(var b=a.next();!b.done;b=a.next())b.value.je("ABORTED")}
xy.prototype.clear=function(){yy(this);this.register.clear()};
var zy=new xy;var Ay=Date.now().toString();
function By(){a:{if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];var d=a;break a}catch(e){}d=Array(16);for(a=0;a<16;a++){b=Date.now();for(c=0;c<b%23;c++)d[a]=Math.random();d[a]=Math.floor(Math.random()*256)}if(Ay)for(a=1,b=0;b<Ay.length;b++)d[a%16]^=d[(a-1)%16]/4^Ay.charCodeAt(b),a++}a=[];for(b=0;b<d.length;b++)a.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(d[b]&63));
return a.join("")}
;var Cy,Dy=D.ytLoggingDocDocumentNonce_;Dy||(Dy=By(),E("ytLoggingDocDocumentNonce_",Dy));Cy=Dy;function Ey(a){this.h=a}
p=Ey.prototype;p.getAsJson=function(){var a={};this.h.trackingParams!==void 0?a.trackingParams=this.h.trackingParams:(a.veType=this.h.veType,this.h.veCounter!==void 0&&(a.veCounter=this.h.veCounter),this.h.elementIndex!==void 0&&(a.elementIndex=this.h.elementIndex));this.h.dataElement!==void 0&&(a.dataElement=this.h.dataElement.getAsJson());this.h.youtubeData!==void 0&&(a.youtubeData=this.h.youtubeData);this.h.isCounterfactual&&(a.isCounterfactual=!0);return a};
p.getAsJspb=function(){var a=new Kp;this.h.trackingParams!==void 0?a.setTrackingParams(this.h.trackingParams):(this.h.veType!==void 0&&eg(a,2,kf(this.h.veType)),this.h.veCounter!==void 0&&eg(a,6,kf(this.h.veCounter)),this.h.elementIndex!==void 0&&eg(a,3,kf(this.h.elementIndex)),this.h.isCounterfactual&&eg(a,5,ff(!0)));if(this.h.dataElement!==void 0){var b=this.h.dataElement.getAsJspb();sg(a,Kp,7,b)}this.h.youtubeData!==void 0&&sg(a,Jp,8,this.h.jspbYoutubeData);return a};
p.toString=function(){return JSON.stringify(this.getAsJson())};
p.isClientVe=function(){return!this.h.trackingParams&&!!this.h.veType};
p.getLoggingDirectives=function(){return this.h.loggingDirectives};function Fy(a){return S("client-screen-nonce-store",{})[a===void 0?0:a]}
function Gy(a,b){b=b===void 0?0:b;var c=S("client-screen-nonce-store");c||(c={},gq("client-screen-nonce-store",c));c[b]=a}
function Hy(a){a=a===void 0?0:a;return a===0?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function Iy(a){return S(Hy(a===void 0?0:a))}
E("yt_logging_screen.getRootVeType",Iy);function Jy(){var a=S("csn-to-ctt-auth-info");a||(a={},gq("csn-to-ctt-auth-info",a));return a}
function Ky(){return Object.values(S("client-screen-nonce-store",{})).filter(function(a){return a!==void 0})}
function Ly(a){a=Fy(a===void 0?0:a);if(!a&&!S("USE_CSN_FALLBACK",!0))return null;a||(a="UNDEFINED_CSN");return a?a:null}
E("yt_logging_screen.getCurrentCsn",Ly);function My(a,b,c){var d=Jy();(c=Ly(c))&&delete d[c];b&&(d[a]=b)}
function Ny(a){return Jy()[a]}
E("yt_logging_screen.getCttAuthInfo",Ny);E("yt_logging_screen.setCurrentScreen",function(a,b,c,d){c=c===void 0?0:c;if(a!==Fy(c)||b!==S(Hy(c)))if(My(a,d,c),Gy(a,c),gq(Hy(c),b),b=function(){setTimeout(function(){a&&Es("foregroundHeartbeatScreenAssociated",{clientDocumentNonce:Cy,clientScreenNonce:a})},0)},"requestAnimationFrame"in window)try{window.requestAnimationFrame(b)}catch(e){b()}else b()});function Oy(){var a=Mi(Py),b;return(new Gk(function(c,d){a.onSuccess=function(e){Cq(e)?c(new Qy(e)):d(new Ry("Request failed, status="+Dq(e),"net.badstatus",e))};
a.onError=function(e){d(new Ry("Unknown request error","net.unknown",e))};
a.onTimeout=function(e){d(new Ry("Request timed out","net.timeout",e))};
b=Tq("//googleads.g.doubleclick.net/pagead/id",a)})).cd(function(c){if(c instanceof Pk){var d;
(d=b)==null||d.abort()}return Lk(c)})}
function Ry(a,b,c){ib.call(this,a+", errorCode="+b);this.errorCode=b;this.xhr=c;this.name="PromiseAjaxError"}
v(Ry,ib);function Qy(a){this.xhr=a}
;function Sy(){this.Z=0;this.h=null}
Sy.prototype.then=function(a,b,c){return this.Z===1&&a?(a=a.call(c,this.h))&&typeof a.then==="function"?a:Ty(a):this.Z===2&&b?(a=b.call(c,this.h))&&typeof a.then==="function"?a:Uy(a):this};
Sy.prototype.getValue=function(){return this.h};
Sy.prototype.isRejected=function(){return this.Z==2};
Sy.prototype.$goog_Thenable=!0;function Uy(a){var b=new Sy;a=a===void 0?null:a;b.Z=2;b.h=a===void 0?null:a;return b}
function Ty(a){var b=new Sy;a=a===void 0?null:a;b.Z=1;b.h=a===void 0?null:a;return b}
;function Vy(a){var b=S("INNERTUBE_HOST_OVERRIDE");b&&(a=String(b)+String(nc(a)));return a}
function Wy(a){var b={};T("json_condensed_response")&&(b.prettyPrint="false");return a=wq(a,b||{},!1)}
function Xy(a,b){var c=c===void 0?{}:c;a={method:b===void 0?"POST":b,mode:xq(a)?"same-origin":"cors",credentials:xq(a)?"same-origin":"include"};b={};for(var d=y(Object.keys(c)),e=d.next();!e.done;e=d.next())e=e.value,c[e]&&(b[e]=c[e]);Object.keys(b).length>0&&(a.headers=b);return a}
;function Yy(){return ui()||(Ad||Bd)&&Ks("applewebkit")&&!Ks("version")&&(!Ks("safari")||Ks("gsa/"))||zd&&Ks("version/")?!0:S("EOM_VISITOR_DATA")?!1:!0}
;function Zy(a){var b=a.docid||a.video_id||a.videoId||a.id;if(b)return b;b=a.raw_player_response;b||(a=a.player_response)&&(b=JSON.parse(a));return b&&b.videoDetails&&b.videoDetails.videoId||null}
;function $y(a){a:{var b="EMBEDDED_PLAYER_MODE_UNKNOWN";window.location.hostname.includes("youtubeeducation.com")&&(b="EMBEDDED_PLAYER_MODE_PFL");var c=a.raw_embedded_player_response;if(!c&&(a=a.embedded_player_response))try{c=JSON.parse(a)}catch(e){break a}if(c)b:for(var d in Op)if(Op[d]==c.embeddedPlayerMode){b=Op[d];break b}}return b==="EMBEDDED_PLAYER_MODE_PFL"}
;function az(a){ib.call(this,a.message||a.description||a.name);this.isMissing=a instanceof bz;this.isTimeout=a instanceof Ry&&a.errorCode=="net.timeout";this.isCanceled=a instanceof Pk}
v(az,ib);az.prototype.name="BiscottiError";function bz(){ib.call(this,"Biscotti ID is missing from server")}
v(bz,ib);bz.prototype.name="BiscottiMissingError";var Py={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},cz=null;function dz(){if(T("disable_biscotti_fetch_entirely_for_all_web_clients"))return Error("Biscotti id fetching has been disabled entirely.");if(!Yy())return Error("User has not consented - not fetching biscotti id.");var a=S("PLAYER_VARS",{});if(Ki(a)=="1")return Error("Biscotti ID is not available in private embed mode");if($y(a))return Error("Biscotti id fetching has been disabled for pfl.")}
function $p(){var a=dz();if(a!==void 0)return Lk(a);cz||(cz=Oy().then(ez).cd(function(b){return fz(2,b)}));
return cz}
function ez(a){a=a.xhr.responseText;if(a.lastIndexOf(")]}'",0)!=0)throw new bz;a=JSON.parse(a.substr(4));if((a.type||1)>1)throw new bz;a=a.id;aq(a);cz=Ty(a);gz(18E5,2);return a}
function fz(a,b){b=new az(b);aq("");cz=Uy(b);a>0&&gz(12E4,a-1);throw b;}
function gz(a,b){Eq(function(){Oy().then(ez,function(c){return fz(b,c)}).cd(Ek)},a)}
function hz(){try{var a=G("yt.ads.biscotti.getId_");return a?a():$p()}catch(b){return Lk(b)}}
;var Ob=pa(["data-"]);function iz(a){a&&(a.dataset?a.dataset[jz()]="true":Pb(a))}
function kz(a){return a?a.dataset?a.dataset[jz()]:a.getAttribute("data-loaded"):null}
var lz={};function jz(){return lz.loaded||(lz.loaded="loaded".replace(/\-([a-z])/g,function(a,b){return b.toUpperCase()}))}
;function mz(a){a=a||{};var b={},c={};this.url=a.url||"";this.args=a.args||Mi(b);this.assets=a.assets||{};this.attrs=a.attrs||Mi(c);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
mz.prototype.clone=function(){var a=new mz,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];Na(c)=="object"?a[b]=Mi(c):a[b]=c}return a};var nz=["att/get"],oz=["share/get_share_panel"],pz=["share/get_web_player_share_panel"],qz=["feedback"],rz=["notification/modify_channel_preference"],sz=["browse/edit_playlist"],tz=["subscription/subscribe"],uz=["subscription/unsubscribe"];var vz=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};E("yt.msgs_",vz);function wz(a){bq(vz,arguments)}
;function xz(a,b,c){yz(a,b,c===void 0?null:c)}
function zz(a){a=Az(a);var b=document.getElementById(a);b&&(Sw(a),b.parentNode.removeChild(b))}
function Bz(a,b){a&&b&&(a=""+Qa(b),(a=Cz[a])&&Qw(a))}
function yz(a,b,c){c=c===void 0?null:c;var d=Az(typeof a==="string"?a:a.toString()),e=document.getElementById(d),f=e&&kz(e),g=e&&!f;f?b&&b():(b&&(f=Ow(d,b),b=""+Qa(b),Cz[b]=f),g||(e=Dz(a,d,function(){kz(e)||(iz(e),Rw(d),Eq(function(){Sw(d)},0))},c)))}
function Dz(a,b,c,d){d=d===void 0?null:d;var e=Ti("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);Lb(e,typeof a==="string"?Fp(a):a);a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function Az(a){var b=document.createElement("a");Db(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+ic(a)}
var Cz={};function Ez(a){var b=Fz(a),c=document.getElementById(b),d=c&&kz(c);d||c&&!d||(c=Gz(a,b,function(){if(!kz(c)){iz(c);Rw(b);var e=Ya(Sw,b);Eq(e,0)}}))}
function Gz(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Fp(a);Rb(d,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function Fz(a){var b=Ti("A");Db(b,new wb(a));a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+ic(a)}
;function Hz(a){var b=C.apply(1,arguments);if(!Iz(a)||b.some(function(d){return!Iz(d)}))throw Error("Only objects may be merged.");
b=y(b);for(var c=b.next();!c.done;c=b.next())Jz(a,c.value)}
function Jz(a,b){for(var c in b)if(Iz(b[c])){if(c in a&&!Iz(a[c]))throw Error("Cannot merge an object into a non-object.");c in a||(a[c]={});Jz(a[c],b[c])}else if(Kz(b[c])){if(c in a&&!Kz(a[c]))throw Error("Cannot merge an array into a non-array.");c in a||(a[c]=[]);Lz(a[c],b[c])}else a[c]=b[c];return a}
function Lz(a,b){b=y(b);for(var c=b.next();!c.done;c=b.next())c=c.value,Iz(c)?a.push(Jz({},c)):Kz(c)?a.push(Lz([],c)):a.push(c);return a}
function Iz(a){return typeof a==="object"&&!Array.isArray(a)}
function Kz(a){return typeof a==="object"&&Array.isArray(a)}
;var Mz="absolute_experiments app conditional_experiments debugcss debugjs expflag forced_experiments pbj pbjreload sbb spf spfreload sr_bns_address sttick".split(" ");
function Nz(a,b){var c=c===void 0?!0:c;var d=S("VALID_SESSION_TEMPDATA_DOMAINS",[]),e=lc(window.location.href);e&&d.push(e);e=lc(a);if(Sb(d,e)>=0||!e&&a.lastIndexOf("/",0)==0)if(d=document.createElement("a"),Db(d,a),a=d.href)if(a=nc(a),a=oc(a))if(c&&!b.csn&&(b.itct||b.ved)&&(b=Object.assign({csn:Ly()},b)),f){var f=parseInt(f,10);isFinite(f)&&f>0&&Oz(a,b,f)}else Oz(a,b)}
function Oz(a,b,c){a=Pz(a);b=b?rc(b):"";c=c||5;Yy()&&ur(a,b,c)}
function Pz(a){for(var b=y(Mz),c=b.next();!c.done;c=b.next())a=wc(a,c.value);return"ST-"+ic(a).toString(36)}
;function Qz(a){for(var b=0,c=0;c<a.length;c++)b=b*31+a.charCodeAt(c),c<a.length-1&&(b%=0x800000000000);return b%1E5}
;function Rz(a){Du.call(this,1,arguments);this.csn=a}
v(Rz,Du);var Mu=new Eu("screen-created",Rz),Sz=[],Tz=0,Uz=new Map,Vz=new Map,Wz=new Map;
function Xz(a,b,c,d,e,f){e=e===void 0?!1:e;f=f===void 0?{}:f;Object.assign(f,Yz({cttAuthInfo:Ny(b)||void 0},b));for(var g=y(d),h=g.next();!h.done;h=g.next()){h=h.value;var k=h.getAsJson();(Ii(k)||!k.trackingParams&&!k.veType)&&sy(Error("Child VE logged with no data"));if(T("no_client_ve_attach_unless_shown")){var l=Zz(h,b);if(k.veType&&!Vz.has(l)&&!Wz.has(l)&&!e){if(!T("il_attach_cache_limit")||Uz.size<1E3){Uz.set(l,[a,b,c,h]);return}T("il_attach_cache_limit")&&Uz.size>1E3&&sy(new U("IL Attach cache exceeded limit"))}h=
Zz(c,b);Uz.has(h)?$z(c,b):Wz.set(h,!0)}}d=d.filter(function(m){m.csn!==b?(m.csn=b,m=!0):m=!1;return m});
c={csn:b,parentVe:c.getAsJson(),childVes:Vb(d,function(m){return m.getAsJson()})};
b==="UNDEFINED_CSN"?aA("visualElementAttached",f,c):a?iy("visualElementAttached",c,a,f):Es("visualElementAttached",c,f)}
function aA(a,b,c){Sz.push({Zf:a,payload:c,hj:void 0,options:b});Tz||(Tz=Nu())}
function Ou(a){if(Sz){for(var b=y(Sz),c=b.next();!c.done;c=b.next())c=c.value,c.payload&&(c.payload.csn=a.csn,Es(c.Zf,c.payload,c.options));Sz.length=0}Tz=0}
function Zz(a,b){return""+a.getAsJson().veType+a.getAsJson().veCounter+b}
function $z(a,b){a=Zz(a,b);if(Uz.has(a)){b=Uz.get(a)||[];var c=c===void 0?{}:c;Xz(b[0],b[1],b[2],[b[3]],!0,c);Uz.delete(a)}}
function Yz(a,b){T("log_sequence_info_on_gel_web")&&(a.sequenceGroup=b);return a}
;function bA(){}
bA.prototype.flush=function(a,b){a=a===void 0?[]:a;b=b===void 0?!1:b;if(T("enable_client_streamz_web")){a=y(a);for(var c=a.next();!c.done;c=a.next())c=ji(c.value),this.h&&sg(c,ei,2,this.h),c={serializedIncrementBatch:Fd(c.j())},Es("streamzIncremented",c,{sendIsolatedPayload:b})}};
function cA(a){var b=new ei;var c=new bi;c=yg(c,1,"botguard");a=yg(c,2,a);a=rg(a,bi);mg(b,1,fi,a);a&&!se(a)&&$f(b.D);this.h=b}
v(cA,bA);var dA=new Map;function eA(){try{return!!self.localStorage}catch(a){return!1}}
;function fA(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function gA(a){if(eA()){var b=Object.keys(window.localStorage);b=y(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=fA(c);d===void 0||a.includes(d)||self.localStorage.removeItem(c)}}}
function hA(){if(!eA())return!1;var a=Mr(),b=Object.keys(window.localStorage);b=y(b);for(var c=b.next();!c.done;c=b.next())if(c=fA(c.value),c!==void 0&&c!==a)return!0;return!1}
;function iA(){var a=!1;try{a=!!window.sessionStorage.getItem("session_logininfo")}catch(b){a=!0}return(S("INNERTUBE_CLIENT_NAME")==="WEB"||S("INNERTUBE_CLIENT_NAME")==="WEB_CREATOR")&&a}
function jA(){var a=a===void 0?!0:a;try{window.sessionStorage.removeItem("stickiness_reload");window.sessionStorage.removeItem("session_logininfo");gq("LOGIN_INFO","");a&&window.sessionStorage.setItem("from_switch_account","1");a=!0;a=a===void 0?!1:a;var b,c=kA;c||(c=document.querySelector("#persist_identity"));if(b=c){var d=b.src?(new URL(b.src)).origin:"*";if(a){var e;(e=b.contentWindow)==null||e.postMessage({action:"clear"},d)}else if(!(Number(window.sessionStorage.getItem("stickiness_reload"))>=
2)){var f=window.sessionStorage.getItem("session_logininfo");if(f){var g;(g=b.contentWindow)==null||g.postMessage({loginInfo:f},d)}}}}catch(h){}}
function lA(a){if(a)if(a.startsWith("https://accounts.google.com/AddSession"))jA();else if(a.startsWith("https://accounts.google.com/ServiceLogin"))jA();else{var b;if(b=a.startsWith("https://myaccount.google.com"))b=(a instanceof $n?a.clone():new $n(a)).h.endsWith("/youtubeoptions");b&&jA()}if(S("LOGGED_IN",!0)&&iA()){b=S("VALID_SESSION_TEMPDATA_DOMAINS",[]);var c=lc(window.location.href);c&&b.push(c);c=lc(a);Sb(b,c)>=0||!c&&a.lastIndexOf("/",0)==0?(b=nc(a),(b=oc(b))?(b=Pz(b),b=(b=vr(b)||null)?tq(b):
{}):b=null):b=null;b==null&&(b={});c=b;var d=void 0;iA()?(d||(d=S("LOGIN_INFO")),d?(c.session_logininfo=d,c=!0):c=!1):c=!1;c&&Nz(a,b)}}
var kA=null;function mA(a,b,c){b=b===void 0?{}:b;c=c===void 0?!1:c;var d=S("EVENT_ID");d&&(b.ei||(b.ei=d));b&&Nz(a,b);if(c)return!1;lA(a);var e=e===void 0?{}:e;var f=f===void 0?"":f;var g=g===void 0?window:g;b=sc(a,e);lA(b);a=void 0;a=a===void 0?Ab:a;a:if(f=b+f,a=a===void 0?Ab:a,!(f instanceof wb)){for(b=0;b<a.length;++b)if(c=a[b],c instanceof yb&&c.If(f)){f=new wb(f);break a}f=void 0}g=g.location;f=Cb(f||xb);f!==void 0&&(g.href=f);return!0}
;function nA(a){if(Ki(S("PLAYER_VARS",{}))!="1"){a&&Zp();try{hz().then(function(){},function(){}),Eq(nA,18E5)}catch(b){lq(b)}}}
;function oA(){this.h={}}
p=oA.prototype;p.contains=function(a){return Object.prototype.hasOwnProperty.call(this.h,a)};
p.get=function(a){if(this.contains(a))return this.h[a]};
p.set=function(a,b){this.h[a]=b};
p.ec=function(){return Object.keys(this.h)};
p.remove=function(a){delete this.h[a]};function pA(){this.mappings=new oA}
pA.prototype.getModuleId=function(a){return a.serviceId.getModuleId()};
pA.prototype.get=function(a){a:{var b=this.mappings.get(a.toString());switch(b.type){case "mapping":a=b.value;break a;case "factory":b=b.value();this.mappings.set(a.toString(),{type:"mapping",value:b});a=b;break a;default:a=Eb(b)}}return a};
new pA;var qA=new Map([["dark","USER_INTERFACE_THEME_DARK"],["light","USER_INTERFACE_THEME_LIGHT"]]);function rA(){var a=a===void 0?window.location.href:a;if(T("kevlar_disable_theme_param"))return null;var b=mc(a);if(T("enable_dark_theme_only_on_shorts")&&b!=null&&b.startsWith("/shorts/"))return"USER_INTERFACE_THEME_DARK";try{var c=uq(a).theme;return qA.get(c)||null}catch(d){}return null}
;function sA(a){var b=new Il;if(a.interpreterJavascript){var c=Dp(a.interpreterJavascript);c=Jb(c).toString();var d=new Gl;yg(d,6,c);sg(b,Gl,1,d,xe)}else a.interpreterUrl&&(c=Ep(a.interpreterUrl),c=qb(c).toString(),d=new Hl,yg(d,4,c),sg(b,Hl,2,d,xe));a.interpreterHash&&lg(b,3,Af(a.interpreterHash),xe);a.program&&lg(b,4,Af(a.program),xe);a.globalName&&lg(b,5,Af(a.globalName),xe);a.clientExperimentsStateBlob&&lg(b,7,Af(a.clientExperimentsStateBlob),xe);return b}
function tA(a){var b={};a=y(a.split("&"));for(var c=a.next();!c.done;c=a.next())c=c.value.split("="),c.length===2&&(b[c[0]]=c[1]);return b}
function uA(a){return Number(a.t)||7200}
;function vA(){var a=a===void 0?window:a;var b,c;return B(function(d){if(d.h==1)return d.yield(Bc(),2);b=a;c=b.bgevmc;if(!c)throw Error("BGE Controls not exposed");return d.return({pause:function(){c.p()},
resume:function(){c.r()},
checkForRefresh:function(){return c.cr()}})})}
function Cc(){if(T("bg_st_hr"))return"havuokmhhs-0";var a,b=((a=globalThis.performance)==null?void 0:a.timeOrigin)||0;return"havuokmhhs-"+Math.floor(b)}
function wA(a){window.bgens=a}
function xA(a){this.h=a}
xA.prototype.bindInnertubeChallengeFetcher=function(a){this.h.bicf(a)};
xA.prototype.registerChallengeFetchedCallback=function(a){this.h.bcr(a)};
xA.prototype.getLatestChallengeResponse=function(){return this.h.blc()};
function yA(){return new Promise(function(a){var b=window;b.ntpevasrs!==void 0?a(new xA(b.ntpevasrs)):(b.ntpqfbel===void 0&&(b.ntpqfbel=[]),b.ntpqfbel.push(function(c){a(new xA(c))}))})}
;var zA=pa(["https://static.doubleclick.net/instream/ad_status.js"]),AA=[],BA=function(a){var b=C.apply(1,arguments);if(b.length===0)return pb(a[0]);for(var c=a[0],d=0;d<b.length;d++)c+=encodeURIComponent(b[d])+a[d+1];return pb(c)}(zA),CA=!1;
function DA(){if(Yy()){var a=S("PLAYER_VARS",{});if(Ki(a)!="1"&&!$y(a)){var b=function(){CA=!0;"google_ad_status"in window?gq("DCLKSTAT",1):gq("DCLKSTAT",2)};
try{xz(BA,b)}catch(c){}AA.push(vn.ra(function(){if(!(CA||"google_ad_status"in window)){try{Bz(BA.toString(),b)}catch(c){}CA=!0;gq("DCLKSTAT",3)}},5E3))}}}
function EA(){var a=Number(S("DCLKSTAT",0));return isNaN(a)?0:a}
;function X(a){this.h=a}
[new X("b.f_"),new X("j.s_"),new X("r.s_"),new X("e.h_"),new X("i.s_"),new X("s.t_"),new X("p.h_"),new X("s.i_"),new X("f.i_"),new X("a.b_"),new X("a.o_"),new X("g.o_"),new X("p.i_"),new X("p.m_"),new X("n.k_"),new X("i.f_"),new X("a.s_"),new X("m.c_"),new X("n.h_"),new X("o.p_"),new X("m.p_"),new X("o.a_"),new X("d.p_"),new X("e.i_")].reduce(function(a,b){a[b.h]=b;return a},{});function FA(a){return G("ytcsi."+(a||"")+"data_")||GA(a)}
function HA(){var a=FA();a.info||(a.info={});return a.info}
function IA(a){a=FA(a);a.metadata||(a.metadata={});return a.metadata}
function JA(a){a=FA(a);a.tick||(a.tick={});return a.tick}
function KA(a){a=FA(a);if(a.gel){var b=a.gel;b.gelInfos||(b.gelInfos={});b.gelTicks||(b.gelTicks={})}else a.gel={gelTicks:{},gelInfos:{}};return a.gel}
function LA(a){a=KA(a);a.gelInfos||(a.gelInfos={});return a.gelInfos}
function MA(a){var b=FA(a).nonce;b||(b=By(),FA(a).nonce=b);return b}
function GA(a){var b={tick:{},info:{}};E("ytcsi."+(a||"")+"data_",b);return b}
;function NA(){var a=G("ytcsi.debug");a||(a=[],E("ytcsi.debug",a),E("ytcsi.reference",{}));return a}
function OA(a){a=a||"";var b=PA();if(b[a])return b[a];var c=NA(),d={timerName:a,info:{},tick:{},span:{},jspbInfo:[]};c.push(d);return b[a]=d}
function PA(){var a=G("ytcsi.reference");if(a)return a;NA();return G("ytcsi.reference")}
;var Y={},QA=(Y.auto_search="LATENCY_ACTION_AUTO_SEARCH",Y.ad_to_ad="LATENCY_ACTION_AD_TO_AD",Y.ad_to_video="LATENCY_ACTION_AD_TO_VIDEO",Y.app_startup="LATENCY_ACTION_APP_STARTUP",Y.browse="LATENCY_ACTION_BROWSE",Y.cast_splash="LATENCY_ACTION_CAST_SPLASH",Y.channel_activity="LATENCY_ACTION_KIDS_CHANNEL_ACTIVITY",Y.channels="LATENCY_ACTION_CHANNELS",Y.chips="LATENCY_ACTION_CHIPS",Y.commerce_transaction="LATENCY_ACTION_COMMERCE_TRANSACTION",Y.direct_playback="LATENCY_ACTION_DIRECT_PLAYBACK",Y.editor=
"LATENCY_ACTION_EDITOR",Y.embed="LATENCY_ACTION_EMBED",Y.embed_no_video="LATENCY_ACTION_EMBED_NO_VIDEO",Y.entity_key_serialization_perf="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF",Y.entity_key_deserialization_perf="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF",Y.explore="LATENCY_ACTION_EXPLORE",Y.favorites="LATENCY_ACTION_FAVORITES",Y.home="LATENCY_ACTION_HOME",Y.inboarding="LATENCY_ACTION_INBOARDING",Y.landing="LATENCY_ACTION_LANDING",Y.learning="LATENCY_ACTION_LEARNING",Y.learning_journey_browse=
"LATENCY_ACTION_LEARNING_JOURNEY_BROWSE",Y.learning_journey_watch="LATENCY_ACTION_LEARNING_JOURNEY_WATCH",Y.library="LATENCY_ACTION_LIBRARY",Y.live="LATENCY_ACTION_LIVE",Y.live_pagination="LATENCY_ACTION_LIVE_PAGINATION",Y.management="LATENCY_ACTION_MANAGEMENT",Y.mini_app="LATENCY_ACTION_MINI_APP_PLAY",Y.notification_settings="LATENCY_ACTION_KIDS_NOTIFICATION_SETTINGS",Y.onboarding="LATENCY_ACTION_ONBOARDING",Y.parent_profile_settings="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS",Y.parent_tools_collection=
"LATENCY_ACTION_PARENT_TOOLS_COLLECTION",Y.parent_tools_dashboard="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD",Y.player_att="LATENCY_ACTION_PLAYER_ATTESTATION",Y.prebuffer="LATENCY_ACTION_PREBUFFER",Y.prefetch="LATENCY_ACTION_PREFETCH",Y.profile_settings="LATENCY_ACTION_KIDS_PROFILE_SETTINGS",Y.profile_switcher="LATENCY_ACTION_LOGIN",Y.projects="LATENCY_ACTION_PROJECTS",Y.reel_watch="LATENCY_ACTION_REEL_WATCH",Y.results="LATENCY_ACTION_RESULTS",Y.red="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",Y.premium=
"LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE",Y.privacy_policy="LATENCY_ACTION_KIDS_PRIVACY_POLICY",Y.review="LATENCY_ACTION_REVIEW",Y.search_overview_answer="LATENCY_ACTION_SEARCH_OVERVIEW_ANSWER",Y.search_ui="LATENCY_ACTION_SEARCH_UI",Y.search_suggest="LATENCY_ACTION_SUGGEST",Y.search_zero_state="LATENCY_ACTION_SEARCH_ZERO_STATE",Y.secret_code="LATENCY_ACTION_KIDS_SECRET_CODE",Y.seek="LATENCY_ACTION_PLAYER_SEEK",Y.settings="LATENCY_ACTION_SETTINGS",Y.store="LATENCY_ACTION_STORE",Y.supervision_dashboard=
"LATENCY_ACTION_KIDS_SUPERVISION_DASHBOARD",Y.tenx="LATENCY_ACTION_TENX",Y.video_preview="LATENCY_ACTION_VIDEO_PREVIEW",Y.video_to_ad="LATENCY_ACTION_VIDEO_TO_AD",Y.watch="LATENCY_ACTION_WATCH",Y.watch_it_again="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN",Y["watch,watch7"]="LATENCY_ACTION_WATCH",Y["watch,watch7_html5"]="LATENCY_ACTION_WATCH",Y["watch,watch7ad"]="LATENCY_ACTION_WATCH",Y["watch,watch7ad_html5"]="LATENCY_ACTION_WATCH",Y.wn_comments="LATENCY_ACTION_LOAD_COMMENTS",Y.ww_rqs="LATENCY_ACTION_WHO_IS_WATCHING",
Y.voice_assistant="LATENCY_ACTION_VOICE_ASSISTANT",Y.cast_load_by_entity_to_watch="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH",Y.networkless_performance="LATENCY_ACTION_NETWORKLESS_PERFORMANCE",Y.gel_compression="LATENCY_ACTION_GEL_COMPRESSION",Y.gel_jspb_serialize="LATENCY_ACTION_GEL_JSPB_SERIALIZE",Y.attestation_challenge_fetch="LATENCY_ACTION_ATTESTATION_CHALLENGE_FETCH",Y);function RA(a){return QA[a]||"LATENCY_ACTION_UNKNOWN"}
;function SA(a,b){Du.call(this,1,arguments);this.timer=b}
v(SA,Du);var TA=new Eu("aft-recorded",SA);E("ytLoggingGelSequenceIdObj_",D.ytLoggingGelSequenceIdObj_||{});var UA=D.ytLoggingLatencyUsageStats_||{};E("ytLoggingLatencyUsageStats_",UA);function VA(){this.h=0}
function WA(){VA.instance||(VA.instance=new VA);return VA.instance}
VA.prototype.tick=function(a,b,c,d){XA(this,"tick_"+a+"_"+b)||Es("latencyActionTicked",{tickName:a,clientActionNonce:b},{timestamp:c,cttAuthInfo:d})};
VA.prototype.info=function(a,b,c){var d=Object.keys(a).join("");XA(this,"info_"+d+"_"+b)||(a=Object.assign({},a),a.clientActionNonce=b,Es("latencyActionInfo",a,{cttAuthInfo:c}))};
VA.prototype.jspbInfo=function(){};
VA.prototype.span=function(a,b,c){var d=Object.keys(a).join("");XA(this,"span_"+d+"_"+b)||(a.clientActionNonce=b,Es("latencyActionSpan",a,{cttAuthInfo:c}))};
function XA(a,b){UA[b]=UA[b]||{count:0};var c=UA[b];c.count++;c.time=W();a.h||(a.h=Or(function(){var d=W(),e;for(e in UA)UA[e]&&d-UA[e].time>6E4&&delete UA[e];a&&(a.h=0)},5E3));
return c.count>5?(c.count===6&&Math.random()*1E5<1&&(c=new U("CSI data exceeded logging limit with key",b.split("_")),b.indexOf("plev")>=0||sy(c)),!0):!1}
;var YA=window;function ZA(){this.timing={};this.clearResourceTimings=function(){};
this.webkitClearResourceTimings=function(){};
this.mozClearResourceTimings=function(){};
this.msClearResourceTimings=function(){};
this.oClearResourceTimings=function(){}}
function $A(){var a;if(T("csi_use_performance_navigation_timing")){var b,c,d,e=Z==null?void 0:(a=Z.getEntriesByType)==null?void 0:(b=a.call(Z,"navigation"))==null?void 0:(c=b[0])==null?void 0:(d=c.toJSON)==null?void 0:d.call(c);e?(e.requestStart=aB(e.requestStart),e.responseEnd=aB(e.responseEnd),e.redirectStart=aB(e.redirectStart),e.redirectEnd=aB(e.redirectEnd),e.domainLookupEnd=aB(e.domainLookupEnd),e.connectStart=aB(e.connectStart),e.connectEnd=aB(e.connectEnd),e.responseStart=aB(e.responseStart),
e.secureConnectionStart=aB(e.secureConnectionStart),e.domainLookupStart=aB(e.domainLookupStart),e.isPerformanceNavigationTiming=!0,a=e):a=Z.timing}else a=T("csi_performance_timing_to_object")?JSON.parse(JSON.stringify(Z.timing)):Z.timing;return a}
function aB(a){return Math.round(bB()+a)}
function bB(){return(T("csi_use_time_origin")||T("csi_use_time_origin_tvhtml5"))&&Z.timeOrigin?Math.floor(Z.timeOrigin):Z.timing.navigationStart}
var Z=YA.performance||YA.mozPerformance||YA.msPerformance||YA.webkitPerformance||new ZA;var cB=!1,dB=!1,eB={'script[name="scheduler/scheduler"]':"sj",'script[name="player/base"]':"pj",'link[rel="preload"][name="player/embed"]':"pej",'link[rel="stylesheet"][name="www-player"]':"pc",'link[rel="stylesheet"][name="player/www-player"]':"pc",'script[name="desktop_polymer/desktop_polymer"]':"dpj",'link[rel="import"][name="desktop_polymer"]':"dph",'script[name="mobile-c3"]':"mcj",'link[rel="stylesheet"][name="mobile-c3"]':"mcc",'script[name="player-plasma-ias-phone/base"]':"mcppj",'script[name="player-plasma-ias-tablet/base"]':"mcptj",
'link[rel="stylesheet"][name="mobile-polymer-player-ias"]':"mcpc",'link[rel="stylesheet"][name="mobile-polymer-player-svg-ias"]':"mcpsc",'script[name="mobile_blazer_core_mod"]':"mbcj",'link[rel="stylesheet"][name="mobile_blazer_css"]':"mbc",'script[name="mobile_blazer_logged_in_users_mod"]':"mbliuj",'script[name="mobile_blazer_logged_out_users_mod"]':"mblouj",'script[name="mobile_blazer_noncore_mod"]':"mbnj","#player_css":"mbpc",'script[name="mobile_blazer_desktopplayer_mod"]':"mbpj",'link[rel="stylesheet"][name="mobile_blazer_tablet_css"]':"mbtc",
'script[name="mobile_blazer_watch_mod"]':"mbwj",'script[name="embed_client"]':"ecj",'link[rel="stylesheet"][name="embed-ui"]':"ecc"},fB=Xa(Z.clearResourceTimings||Z.webkitClearResourceTimings||Z.mozClearResourceTimings||Z.msClearResourceTimings||Z.oClearResourceTimings||Ek,Z);function gB(a,b){hB("_start",a,b)}
function iB(a,b){if(!T("web_csi_action_sampling_enabled")||!FA(b).actionDisabled){var c=OA(b||"");Hz(c.info,a);a.loadType&&(c=a.loadType,IA(b).loadType=c);Hz(LA(b),a);c=MA(b);b=FA(b).cttAuthInfo;WA().info(a,c,b)}}
function jB(){var a,b,c,d;return((d=bx().resolve(new Ww(qu))==null?void 0:(a=ru())==null?void 0:(b=a.loggingHotConfig)==null?void 0:(c=b.csiConfig)==null?void 0:c.debugTicks)!=null?d:[]).map(function(e){return Object.values(e)[0]})}
function hB(a,b,c){if(!T("web_csi_action_sampling_enabled")||!FA(c).actionDisabled){var d=MA(c),e;if(e=T("web_csi_debug_sample_enabled")&&d){(bx().resolve(new Ww(qu))==null?0:ru())&&!dB&&(dB=!0,hB("gcfl",W(),c));var f,g,h;e=(bx().resolve(new Ww(qu))==null?void 0:(f=ru())==null?void 0:(g=f.loggingHotConfig)==null?void 0:(h=g.csiConfig)==null?void 0:h.debugSampleWeight)||0;if(f=e!==0)b:{f=jB();if(f.length>0)for(g=0;g<f.length;g++)if(a===f[g]){f=!0;break b}f=!1}f?(e=Qz(d)%e!==0,FA(c).debugTicksExcludedLogged||
(f={},f.debugTicksExcluded=e,iB(f,c)),FA(c).debugTicksExcludedLogged=!0):e=!1}if(!e){if(a[0]!=="_"&&(e=a,f=b,Z.mark))if(e.startsWith("mark_")||(e="mark_"+e),c&&(e+=" ("+c+")"),f===void 0||T("web_csi_disable_alt_time_performance_mark"))Z.mark(e);else{f=T("csi_use_performance_navigation_timing")?f-Z.timeOrigin:f-(Z.timeOrigin||Z.timing.navigationStart);try{Z.mark(e,{startTime:f})}catch(k){}}e=OA(c||"");e.tick[a]=b||W();if(e.callback&&e.callback[a])for(e=y(e.callback[a]),f=e.next();!f.done;f=e.next())f=
f.value,f();e=KA(c);e.gelTicks&&(e.gelTicks[a]=!0);f=JA(c);e=b||W();f[a]=e;f=FA(c).cttAuthInfo;a==="_start"?(a=WA(),XA(a,"baseline_"+d)||Es("latencyActionBaselined",{clientActionNonce:d},{timestamp:b,cttAuthInfo:f})):WA().tick(a,d,b,f);kB(c);return e}}}
function lB(){var a,b=(a=Z.getEntriesByType)==null?void 0:a.call(Z,"mark");b&&b.forEach(function(c){if(c.name.startsWith("mark_")){var d;(d=Z.clearMarks)==null||d.call(Z,c.name)}})}
function mB(){var a=document;if("visibilityState"in a)a=a.visibilityState;else{var b=uw+"VisibilityState";a=b in a?a[b]:void 0}switch(a){case "hidden":return 0;case "visible":return 1;case "prerender":return 2;case "unloaded":return 3;default:return-1}}
function nB(){function a(f,g,h){g=g.match("_rid")?g.split("_rid")[0]:g;typeof h==="number"&&(h=JSON.stringify(h));f.requestIds?f.requestIds.push({endpoint:g,id:h}):f.requestIds=[{endpoint:g,id:h}]}
for(var b={},c=y(Object.entries(S("TIMING_INFO",{}))),d=c.next();!d.done;d=c.next()){var e=y(d.value);d=e.next().value;e=e.next().value;switch(d){case "GetBrowse_rid":a(b,d,e);break;case "GetGuide_rid":a(b,d,e);break;case "GetHome_rid":a(b,d,e);break;case "GetPlayer_rid":a(b,d,e);break;case "GetSearch_rid":a(b,d,e);break;case "GetSettings_rid":a(b,d,e);break;case "GetTrending_rid":a(b,d,e);break;case "GetWatchNext_rid":a(b,d,e);break;case "yt_red":b.isRedSubscriber=!!e;break;case "yt_ad":b.isMonetized=
!!e}}return b}
function oB(a,b){a=document.querySelector(a);if(!a)return!1;var c="",d=a.nodeName;d==="SCRIPT"?(c=a.src,c||(c=a.getAttribute("data-timing-href"))&&(c=window.location.protocol+c)):d==="LINK"&&(c=a.href);Gb(document)&&a.setAttribute("nonce",Gb(document));return c?(a=Z.getEntriesByName(c))&&a[0]&&(a=a[0],c=bB(),hB("rsf_"+b,c+Math.round(a.fetchStart)),hB("rse_"+b,c+Math.round(a.responseEnd)),a.transferSize!==void 0&&a.transferSize===0)?!0:!1:!1}
function pB(){var a=window.location.protocol,b=Z.getEntriesByType("resource");b=Ub(b,function(c){return c.name.indexOf(a+"//fonts.gstatic.com/s/")===0});
(b=Wb(b,function(c,d){return d.duration>c.duration?d:c},{duration:0}))&&b.startTime>0&&b.responseEnd>0&&(hB("wffs",aB(b.startTime)),hB("wffe",aB(b.responseEnd)))}
function qB(a){var b=rB("aft",a);if(b)return b;b=S((a||"")+"TIMING_AFT_KEYS",["ol"]);for(var c=b.length,d=0;d<c;d++){var e=rB(b[d],a);if(e)return e}return NaN}
function sB(a){E("ytglobal.timing"+(a||"")+"ready_",!0)}
function rB(a,b){if(a=JA(b)[a])return typeof a==="number"?a:a[a.length-1]}
function kB(a){var b=rB("_start",a),c=qB(a),d=!cB;b&&c&&d&&(Ju(TA,new SA(Math.round(c-b),a)),cB=!0)}
function tB(){if(Z.getEntriesByType){var a=Z.getEntriesByType("paint");if(a=Xb(a,function(c){return c.name==="first-paint"}))return aB(a.startTime)}var b;
T("csi_use_performance_navigation_timing")?b=Z.getEntriesByType("first-paint")[0].startTime:b=Z.timing.pj;return b?Math.max(0,b):0}
;function uB(a,b){kq(function(){OA("").info.actionType=a;b&&gq("TIMING_AFT_KEYS",b);gq("TIMING_ACTION",a);var c=nB();Object.keys(c).length>0&&iB(c);c={isNavigation:!0,actionType:RA(S("TIMING_ACTION"))};var d=S("PREVIOUS_ACTION");d&&(c.previousAction=RA(d));if(d=S("CLIENT_PROTOCOL"))c.httpProtocol=d;if(d=S("CLIENT_TRANSPORT"))c.transportProtocol=d;(d=Ly())&&d!=="UNDEFINED_CSN"&&(c.clientScreenNonce=d);d=mB();if(d===1||d===-1)c.isVisible=!0;IA();HA();c.loadType="cold";d=HA();var e=$A(),f=bB(),g=S("CSI_START_TIMESTAMP_MILLIS",
0);g>0&&!T("embeds_web_enable_csi_start_override_killswitch")&&(f=g);f&&(hB("srt",e.responseStart),d.prerender!==1&&gB(f));d=tB();d>0&&hB("fpt",d);d=$A();d.isPerformanceNavigationTiming&&iB({performanceNavigationTiming:!0},void 0);hB("nreqs",d.requestStart,void 0);hB("nress",d.responseStart,void 0);hB("nrese",d.responseEnd,void 0);d.redirectEnd-d.redirectStart>0&&(hB("nrs",d.redirectStart,void 0),hB("nre",d.redirectEnd,void 0));d.domainLookupEnd-d.domainLookupStart>0&&(hB("ndnss",d.domainLookupStart,
void 0),hB("ndnse",d.domainLookupEnd,void 0));d.connectEnd-d.connectStart>0&&(hB("ntcps",d.connectStart,void 0),hB("ntcpe",d.connectEnd,void 0));d.secureConnectionStart>=bB()&&d.connectEnd-d.secureConnectionStart>0&&(hB("nstcps",d.secureConnectionStart,void 0),hB("ntcpe",d.connectEnd,void 0));Z&&"getEntriesByType"in Z&&pB();d=[];if(document.querySelector&&Z&&Z.getEntriesByName)for(var h in eB)eB.hasOwnProperty(h)&&(e=eB[h],oB(h,e)&&d.push(e));if(d.length>0)for(c.resourceInfo=[],h=y(d),d=h.next();!d.done;d=
h.next())c.resourceInfo.push({resourceCache:d.value});iB(c);c=KA();c.preLoggedGelInfos||(c.preLoggedGelInfos=[]);h=c.preLoggedGelInfos;c=LA();d=void 0;for(e=0;e<h.length;e++)if(f=h[e],f.loadType){d=f.loadType;break}if(IA().loadType==="cold"&&(c.loadType==="cold"||d==="cold")){d=JA();e=KA();e=e.gelTicks?e.gelTicks:e.gelTicks={};for(var k in d)k in e||typeof d[k]==="number"&&hB(k,rB(k));k={};d=!1;h=y(h);for(e=h.next();!e.done;e=h.next())d=e.value,Hz(c,d),Hz(k,d),d=!0;d&&iB(k)}sB();k=S("TIMING_ACTION");
G("ytglobal.timingready_")&&k&&vB()&&qB()&&kB()})()}
function wB(){var a=a===void 0?{}:a;kq(function(){xB();var b=a.sampleRate;if(!T("web_csi_action_sampling_enabled")||b===void 0||b<=1)b=!1;else{var c=MA("attestation_challenge_fetch");b=Qz(c)%b!==0}b&&(FA("attestation_challenge_fetch").actionDisabled=!0);OA("attestation_challenge_fetch").info.actionType="attestation_challenge_fetch";a.cttAuthInfo&&(FA("attestation_challenge_fetch").cttAuthInfo=a.cttAuthInfo);gq("attestation_challenge_fetchTIMING_ACTION","attestation_challenge_fetch");kq(gB)(a.startTime,
"attestation_challenge_fetch");b={actionType:RA("attestation_challenge_fetch")};a.lj&&(b.previousAction=RA(S("TIMING_ACTION")));(c=Ly())&&c!=="UNDEFINED_CSN"&&(b.clientScreenNonce=c);yB(b,"attestation_challenge_fetch");sB("attestation_challenge_fetch")})()}
function xB(){kq(function(){vB("attestation_challenge_fetch")&&zB("aa",void 0,"attestation_challenge_fetch");var a=PA();a.attestation_challenge_fetch&&delete a.attestation_challenge_fetch;var b={timerName:"attestation_challenge_fetch",info:{},tick:{},span:{},jspbInfo:[]};NA().push(b);a.attestation_challenge_fetch=b;GA("attestation_challenge_fetch");fB();lB()})()}
function vB(a){return kq(function(){return AB("_start",a)})()}
function yB(a,b,c){kq(iB)(a,b,c===void 0?!1:c)}
function zB(a,b,c){return kq(hB)(a,b,c)}
function AB(a,b){return kq(function(){var c=JA(b);return a in c})()}
function BB(a){if(!T("universal_csi_network_ticks"))return"";a=mc(a)||"";for(var b=Object.keys(Bu),c=0;c<b.length;c++){var d=b[c];if(a.includes(d))return d}return""}
function CB(a){if(!T("universal_csi_network_ticks"))return function(){};
var b=Bu[a];return b?(DB(b),function(){var c=T("universal_csi_network_ticks")?(c=Cu[a])?DB(c):!1:!1;return c}):function(){}}
function DB(a){return kq(function(){if(AB(a))return!1;zB(a,void 0,void 0);return!0})()}
function EB(a){kq(function(){if(!vB("attestation_challenge_fetch")||AB(a,"attestation_challenge_fetch"))return!1;zB(a,void 0,"attestation_challenge_fetch");return!0})()}
function FB(){kq(function(){var a=MA();requestAnimationFrame(function(){setTimeout(function(){a===MA()&&zB("ol",void 0,void 0)},0)})})()}
var GB=window;GB.ytcsi&&(GB.ytcsi.infoGel=yB,GB.ytcsi.tick=zB);function HB(a,b,c){var d=this;this.network=a;this.options=b;this.o=c;this.h=null;if(b.ye){var e=new Ml;this.h=e.promise;D.ytAtRC&&vn.Xa(function(){var f,g;return B(function(h){if(h.h==1){if(!D.ytAtRC)return h.return();f=IB(null);return h.yield(d.Db(f),2)}g=h.i;D.ytAtRC&&D.ytAtRC(JSON.stringify(g));h.h=0})},2);
yA().then(function(f){var g,h,k,l;return B(function(m){if(m.h==1)return f.bindInnertubeChallengeFetcher(function(n){return d.Db(IB(n))}),m.yield(Bc(),2);
g=m.i;h=f.getLatestChallengeResponse();k=h.challenge;if(!k)throw Error("BGE_MACIL");l={challenge:k,mb:tA(k),vm:g,bgChallenge:new Il};e.resolve(l);f.registerChallengeFetchedCallback(function(n){n=n.challenge;if(!n)throw Error("BGE_MACR");n={challenge:n,mb:tA(n),vm:g,bgChallenge:new Il};d.h=Promise.resolve(n)});
m.h=0})})}else b.preload&&JB(this,new Promise(function(f){Or(function(){f(KB(d))},0)}))}
HB.prototype.j=function(){var a=this;return B(function(b){return b.h==1?b.yield(Promise.race([a.h,null]),2):b.return(!!b.i)})};
HB.prototype.i=function(a,b,c){var d=this,e,f,g;return B(function(h){d.h===null&&JB(d,KB(d));e=!1;f={};g=function(){var k,l,m,n,r;return B(function(t){switch(t.h){case 1:if(!d.options.aj||!d.options.ye){t.v(2);break}return t.yield(vA(),3);case 3:return k=t.i,t.yield((l=k)==null?void 0:l.checkForRefresh(),2);case 2:return t.yield(d.h,5);case 5:m=t.i;f.challenge=m.challenge;if(!m.vm){"c1a"in m.mb&&(f.error="ATTESTATION_ERROR_VM_NOT_INITIALIZED");t.v(6);break}n=Object.assign({},{c:m.challenge,e:a},b);
wa(t,7);e=!0;return t.yield(m.vm.snapshot({Ha:n}),9);case 9:(r=t.i)?f.webResponse=r:f.error="ATTESTATION_ERROR_VM_NO_RESPONSE";xa(t,6);break;case 7:ya(t),f.error="ATTESTATION_ERROR_VM_INTERNAL_ERROR";case 6:if(a==="ENGAGEMENT_TYPE_PLAYBACK"){var w=m.mb,z={};w.c6a&&(z.reportingStatus=String(Number(w.c)^EA()));w.c6b&&(z.broadSpectrumDetectionResult=String(Number(w.c)^Number(S("CATSTAT",0))));f.adblockReporting=z}return t.return(f)}})};
return h.return(Promise.race([g(),LB(c,function(){var k=Object.assign({},f);e&&(k.error="ATTESTATION_ERROR_VM_TIMEOUT");return k})]))})};
function IB(a){var b={engagementType:"ENGAGEMENT_TYPE_UNBOUND"};a&&(b.interpreterHash=a);return b}
function KB(a,b){b=b===void 0?0:b;var c,d,e,f,g,h,k,l,m,n,r,t;return B(function(w){switch(w.h){case 1:c=IB(Rl().h);if(T("att_fet_ks"))return wa(w,7),w.yield(a.Db(c),9);wa(w,4);return w.yield(MB(a,c),6);case 6:g=w.i;e=g.Wf;f=g.Xf;d=g;xa(w,3);break;case 4:return ya(w),sy(Error("Failed to fetch attestation challenge after "+(b+" attempts; not retrying for 24h."))),NB(a,864E5),w.return({challenge:"",mb:{},vm:void 0,bgChallenge:void 0});case 9:d=w.i;if(!d)throw Error("Fetching Attestation challenge returned falsy");
if(!d.challenge)throw Error("Missing Attestation challenge");e=d.challenge;f=tA(e);if("c1a"in f&&(!d.bgChallenge||!d.bgChallenge.program))throw Error("Expected bg challenge but missing.");xa(w,3);break;case 7:h=ya(w);sy(h);b++;if(b>=5)return sy(Error("Failed to fetch attestation challenge after "+(b+" attempts; not retrying for 24h."))),NB(a,864E5),w.return({challenge:"",mb:{},vm:void 0,bgChallenge:void 0});k=1E3*Math.pow(2,b-1)+Math.random()*1E3;return w.return(new Promise(function(z){Or(function(){z(KB(a,
b))},k)}));
case 3:l=uA(f);NB(a,l*1E3);m=void 0;if(!("c1a"in f&&d.bgChallenge)){w.v(10);break}n=sA(d.bgChallenge);wa(w,11);return w.yield(Sl(Rl(),n),13);case 13:xa(w,12);break;case 11:return r=ya(w),sy(r),w.return({challenge:e,mb:f,vm:m,bgChallenge:n});case 12:return wa(w,14),m=new Ol({challenge:n,Lb:{qa:"aGIf"}}),w.yield(m.Pb,16);case 16:xa(w,10);break;case 14:t=ya(w),sy(t),m=void 0;case 10:return w.return({challenge:e,mb:f,vm:m,bgChallenge:n})}})}
function OB(a,b){var c;return B(function(d){if(d.h==1)return wA(2),wa(d,2),d.yield(a.network.Db(b),4);if(d.h!=2)return(c=d.i)?c.challenge&&!c.bgChallenge?wA(1):wA(4):wA(3),d.return(c);ya(d);wA(3);return d.return(void 0)})}
HB.prototype.Db=function(a){var b=this,c;return B(function(d){c=b.o;if(!c||c.va())return d.return(OB(b,a));EB("att_pna");return d.return(new Promise(function(e){lk(c,"publicytnetworkstatus-online",function(){OB(b,a).then(e)})}))})};
function PB(a){if(!a)throw Error("Fetching Attestation challenge returned falsy");if(!a.challenge)throw Error("Missing Attestation challenge");var b=a.challenge,c=tA(b);if("c1a"in c&&(!a.bgChallenge||!a.bgChallenge.program))throw Error("Expected bg challenge but missing.");return Object.assign({},a,{Wf:b,Xf:c})}
function MB(a,b){var c,d,e,f,g;return B(function(h){switch(h.h){case 1:c=void 0,d=0,e={};case 2:if(!(d<5)){h.v(4);break}if(!(d>0)){h.v(5);break}e.Md=1E3*Math.pow(2,d-1)+Math.random()*1E3;return h.yield(new Promise(function(k){return function(l){Or(function(){l(void 0)},k.Md)}}(e)),5);
case 5:return wa(h,7),h.yield(a.Db(b),9);case 9:return f=h.i,h.return(PB(f));case 7:c=g=ya(h),g instanceof Error&&sy(g);case 8:d++;e={Md:void 0};h.v(2);break;case 4:throw c;}})}
function JB(a,b){a.h=b}
function QB(a){var b,c,d;return B(function(e){if(e.h==1)return e.yield(Promise.race([a.h,null]),2);b=e.i;var f=KB(a);a.h=f;(c=b)==null||(d=c.vm)==null||d.dispose();e.h=0})}
function NB(a,b){function c(){var e;return B(function(f){e=d-Date.now();return e<1E3?f.yield(QB(a),0):(Or(c,Math.min(e,6E4)),f.v(0))})}
var d=Date.now()+b;c()}
function LB(a,b){return new Promise(function(c){Or(function(){c(b())},a)})}
;var RB={},SB=(RB.WEB_UNPLUGGED="^unplugged/",RB.WEB_UNPLUGGED_ONBOARDING="^unplugged/",RB.WEB_UNPLUGGED_OPS="^unplugged/",RB.WEB_UNPLUGGED_PUBLIC="^unplugged/",RB.WEB_CREATOR="^creator/",RB.WEB_KIDS="^kids/",RB.WEB_EXPERIMENTS="^experiments/",RB.WEB_MUSIC="^music/",RB.WEB_REMIX="^music/",RB.WEB_MUSIC_EMBEDDED_PLAYER="^music/",RB.WEB_MUSIC_EMBEDDED_PLAYER="^main_app/|^sfv/",RB);
function TB(a){var b=b===void 0?"UNKNOWN_INTERFACE":b;if(a.length===1)return a[0];var c=SB[b];if(c){c=new RegExp(c);for(var d=y(a),e=d.next();!e.done;e=d.next())if(e=e.value,c.exec(e))return e}var f=[];Object.entries(SB).forEach(function(g){var h=y(g);g=h.next().value;h=h.next().value;b!==g&&f.push(h)});
c=new RegExp(f.join("|"));a.sort(function(g,h){return g.length-h.length});
d=y(a);for(e=d.next();!e.done;e=d.next())if(e=e.value,!c.exec(e))return e;return a[0]}
;function UB(){this.h=VB.instance}
UB.prototype.Db=function(a){EB("att_fsr");return WB(this.h,a).then(function(b){EB("att_frr");return b})};var XB=new Vw("INNERTUBE_TRANSPORT_TOKEN");function YB(){var a,b,c;return B(function(d){if(d.h==1)return a=bx().resolve(XB),a?d.yield(ZB(a),2):(sy(Error("InnertubeTransportService unavailable in fetchDatasyncIds")),d.return(void 0));if(b=d.i){if(b.errorMetadata)return sy(Error("Datasync IDs fetch responded with "+b.errorMetadata.status+": "+b.error)),d.return(void 0);c=b.Wi;return d.return(c)}sy(Error("Network request to get Datasync IDs failed."));return d.return(void 0)})}
;function $B(){this.h={};if(this.i=xr()){var a=vr("CONSISTENCY");a&&aC(this,{encryptedTokenJarContents:a})}}
$B.prototype.handleResponse=function(a,b){if(!b)throw Error("request needs to be passed into ConsistencyService");var c,d;b=((c=b.Ja.context)==null?void 0:(d=c.request)==null?void 0:d.consistencyTokenJars)||[];var e;if(a=(e=a.responseContext)==null?void 0:e.consistencyTokenJar){e=y(b);for(c=e.next();!c.done;c=e.next())delete this.h[c.value.encryptedTokenJarContents];aC(this,a)}};
function aC(a,b){if(b.encryptedTokenJarContents&&(a.h[b.encryptedTokenJarContents]=b,typeof b.expirationSeconds==="string")){var c=Number(b.expirationSeconds);setTimeout(function(){delete a.h[b.encryptedTokenJarContents]},c*1E3);
a.i&&ur("CONSISTENCY",b.encryptedTokenJarContents,c,void 0,!0)}}
;var bC=window.location.hostname.split(".").slice(-2).join(".");function cC(){this.i=-1;var a=S("LOCATION_PLAYABILITY_TOKEN");S("INNERTUBE_CLIENT_NAME")==="TVHTML5"&&(this.localStorage=dC(this))&&(a=this.localStorage.get("yt-location-playability-token"));a&&(this.locationPlayabilityToken=a,this.h=void 0)}
var eC;function fC(){eC=G("yt.clientLocationService.instance");eC||(eC=new cC,E("yt.clientLocationService.instance",eC));return eC}
p=cC.prototype;
p.setLocationOnInnerTubeContext=function(a){a.client||(a.client={});if(this.h)a.client.locationInfo||(a.client.locationInfo={}),a.client.locationInfo.latitudeE7=Math.floor(this.h.coords.latitude*1E7),a.client.locationInfo.longitudeE7=Math.floor(this.h.coords.longitude*1E7),a.client.locationInfo.horizontalAccuracyMeters=Math.round(this.h.coords.accuracy),a.client.locationInfo.forceLocationPlayabilityTokenRefresh=!0;else if(this.j||this.locationPlayabilityToken)a.client.locationPlayabilityToken=this.j||
this.locationPlayabilityToken};
p.handleResponse=function(a){var b;a=(b=a.responseContext)==null?void 0:b.locationPlayabilityToken;a!==void 0&&(this.locationPlayabilityToken=a,this.h=void 0,S("INNERTUBE_CLIENT_NAME")==="TVHTML5"?(this.localStorage=dC(this))&&this.localStorage.set("yt-location-playability-token",a,15552E3):ur("YT_CL",JSON.stringify({loctok:a}),15552E3,bC,!0))};
function dC(a){return a.localStorage===void 0?new vs("yt-client-location"):a.localStorage}
p.clearLocationPlayabilityToken=function(a){a==="TVHTML5"?(this.localStorage=dC(this))&&this.localStorage.remove("yt-location-playability-token"):wr("YT_CL");this.j=void 0;this.i!==-1&&(clearTimeout(this.i),this.i=-1)};
p.getCurrentPositionFromGeolocation=function(){var a=this;if(!(navigator&&navigator.geolocation&&navigator.geolocation.getCurrentPosition))return Promise.reject(Error("Geolocation unsupported"));var b=!1,c=1E4;S("INNERTUBE_CLIENT_NAME")==="MWEB"&&(b=!0,c=15E3);return new Promise(function(d,e){navigator.geolocation.getCurrentPosition(function(f){a.h=f;d(f)},function(f){e(f)},{enableHighAccuracy:b,
maximumAge:0,timeout:c})})};
p.createUnpluggedLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);if(a==null?0:a.accuracy)b.locationRadiusMeters=Math.round(a.accuracy);return b};
p.createLocationInfo=function(a){var b={};a=a.coords;if(a==null?0:a.latitude)b.latitudeE7=Math.floor(a.latitude*1E7);if(a==null?0:a.longitude)b.longitudeE7=Math.floor(a.longitude*1E7);return b};function gC(a,b,c){b=b===void 0?!1:b;c=c===void 0?!1:c;var d=S("INNERTUBE_CONTEXT");if(!d)return ry(Error("Error: No InnerTubeContext shell provided in ytconfig.")),{};d=Ni(d);T("web_no_tracking_params_in_shell_killswitch")||delete d.clickTracking;d.client||(d.client={});var e=d.client;e.clientName==="MWEB"&&e.clientFormFactor!=="AUTOMOTIVE_FORM_FACTOR"&&(e.clientFormFactor=S("IS_TABLET")?"LARGE_FORM_FACTOR":"SMALL_FORM_FACTOR");e.screenWidthPoints=window.innerWidth;e.screenHeightPoints=window.innerHeight;
e.screenPixelDensity=Math.round(window.devicePixelRatio||1);e.screenDensityFloat=window.devicePixelRatio||1;e.utcOffsetMinutes=-Math.floor((new Date).getTimezoneOffset());var f=f===void 0?!1:f;Br();var g="USER_INTERFACE_THEME_LIGHT";Er(165)?g="USER_INTERFACE_THEME_DARK":Er(174)?g="USER_INTERFACE_THEME_LIGHT":!T("kevlar_legacy_browsers")&&window.matchMedia&&window.matchMedia("(prefers-color-scheme)").matches&&window.matchMedia("(prefers-color-scheme: dark)").matches&&(g="USER_INTERFACE_THEME_DARK");
f=f?g:rA()||g;e.userInterfaceTheme=f;if(!b){if(f=Jr())e.connectionType=f;T("web_log_effective_connection_type")&&(f=Kr())&&(d.client.effectiveConnectionType=f)}var h;if(T("web_log_memory_total_kbytes")&&((h=D.navigator)==null?0:h.deviceMemory)){var k;h=(k=D.navigator)==null?void 0:k.deviceMemory;d.client.memoryTotalKbytes=""+h*1E6}T("web_gcf_hashes_innertube")&&(f=uu())&&(k=f.coldConfigData,h=f.coldHashData,f=f.hotHashData,d.client.configInfo=d.client.configInfo||{},k&&(d.client.configInfo.coldConfigData=
k),h&&(d.client.configInfo.coldHashData=h),f&&(d.client.configInfo.hotHashData=f));k=uq(D.location.href);!T("web_populate_internal_geo_killswitch")&&k.internalcountrycode&&(e.internalGeo=k.internalcountrycode);e.clientName==="MWEB"||e.clientName==="WEB"?(e.mainAppWebInfo||(e.mainAppWebInfo={}),e.mainAppWebInfo.graftUrl=D.location.href,T("kevlar_woffle")&&pr.instance&&(k=pr.instance,e.mainAppWebInfo.pwaInstallabilityStatus=!k.h&&k.i?"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED":"PWA_INSTALLABILITY_STATUS_UNKNOWN"),
e.mainAppWebInfo.webDisplayMode=qr(),e.mainAppWebInfo.isWebNativeShareAvailable=navigator&&navigator.share!==void 0):e.clientName==="TVHTML5"&&(!T("web_lr_app_quality_killswitch")&&(k=S("LIVING_ROOM_APP_QUALITY"))&&(e.tvAppInfo=Object.assign(e.tvAppInfo||{},{appQuality:k})),k=S("LIVING_ROOM_CERTIFICATION_SCOPE"))&&(e.tvAppInfo=Object.assign(e.tvAppInfo||{},{certificationScope:k}));if(!T("web_populate_time_zone_itc_killswitch")){a:{if(typeof Intl!=="undefined")try{var l=(new Intl.DateTimeFormat).resolvedOptions().timeZone;
break a}catch(na){}l=void 0}l&&(e.timeZone=l)}(l=S("EXPERIMENTS_TOKEN",""))?e.experimentsToken=l:delete e.experimentsToken;l=Lq();$B.instance||($B.instance=new $B);d.request=Object.assign({},d.request,{internalExperimentFlags:l,consistencyTokenJars:Fi($B.instance.h)});!T("web_prequest_context_killswitch")&&(l=S("INNERTUBE_CONTEXT_PREQUEST_CONTEXT"))&&(d.request.externalPrequestContext=l);e=Br();l=Er(58);e=e.get("gsml","");d.user=Object.assign({},d.user);l&&(d.user.enableSafetyMode=l);e&&(d.user.lockedSafetyMode=
!0);T("warm_op_csn_cleanup")?c&&(b=Ly())&&(d.clientScreenNonce=b):!b&&(b=Ly())&&(d.clientScreenNonce=b);a&&(d.clickTracking={clickTrackingParams:a});if(a=G("yt.mdx.remote.remoteClient_"))d.remoteClient=a;fC().setLocationOnInnerTubeContext(d);try{var m=yq(),n=m.bid;delete m.bid;d.adSignalsInfo={params:[],bid:n};for(var r=y(Object.entries(m)),t=r.next();!t.done;t=r.next()){var w=y(t.value),z=w.next().value,x=w.next().value;m=z;n=x;a=void 0;(a=d.adSignalsInfo.params)==null||a.push({key:m,value:""+n})}var H,
F;if(((H=d.client)==null?void 0:H.clientName)==="TVHTML5"||((F=d.client)==null?void 0:F.clientName)==="TVHTML5_UNPLUGGED"){var V=S("INNERTUBE_CONTEXT");V.adSignalsInfo&&(d.adSignalsInfo.advertisingId=V.adSignalsInfo.advertisingId,d.adSignalsInfo.advertisingIdSignalType="DEVICE_ID_TYPE_CONNECTED_TV_IFA",d.adSignalsInfo.limitAdTracking=V.adSignalsInfo.limitAdTracking)}}catch(na){ry(na)}return d}
;function hC(a){var b={"Content-Type":"application/json"};S("EOM_VISITOR_DATA")?b["X-Goog-EOM-Visitor-Id"]=S("EOM_VISITOR_DATA"):S("VISITOR_DATA")&&(b["X-Goog-Visitor-Id"]=S("VISITOR_DATA"));b["X-Youtube-Bootstrap-Logged-In"]=S("LOGGED_IN",!1);S("DEBUG_SETTINGS_METADATA")&&(b["X-Debug-Settings-Metadata"]=S("DEBUG_SETTINGS_METADATA"));a!=="cors"&&((a=S("INNERTUBE_CONTEXT_CLIENT_NAME"))&&(b["X-Youtube-Client-Name"]=a),(a=S("INNERTUBE_CONTEXT_CLIENT_VERSION"))&&(b["X-Youtube-Client-Version"]=a),(a=S("CHROME_CONNECTED_HEADER"))&&
(b["X-Youtube-Chrome-Connected"]=a),(a=S("DOMAIN_ADMIN_STATE"))&&(b["X-Youtube-Domain-Admin-State"]=a));(a=S("SERIALIZED_LAVA_DEVICE_CONTEXT"))&&(b["X-YouTube-Lava-Device-Context"]=a);return b}
;function iC(a){return function(){return new a}}
;function jC(){}
jC.prototype.u=function(a,b,c){b=b===void 0?{}:b;c=c===void 0?tr:c;var d={context:gC(a.clickTrackingParams,!1,this.o)};var e=this.i(a);if(e){this.h(d,e,b);var f;b="/youtubei/v1/"+TB(this.j());(e=(f=Ax(a.commandMetadata,Mp))==null?void 0:f.apiUrl)&&(b=e);f=Wy(Vy(b));a=Object.assign({},{command:a},void 0);d={input:f,hb:Xy(f),Ja:d,config:a};d.config.Yb?d.config.Yb.identity=c:d.config.Yb={identity:c};return d}c=new U("Error: Failed to create Request from Command.",a);ry(c)};
da.Object.defineProperties(jC.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!1}}});
function kC(){}
v(kC,jC);function lC(){}
v(lC,kC);lC.prototype.u=function(){return{input:"/getDatasyncIdsEndpoint",hb:Xy("/getDatasyncIdsEndpoint","GET"),Ja:{}}};
lC.prototype.j=function(){return[]};
lC.prototype.i=function(){};
lC.prototype.h=function(){};var mC={},nC=(mC.GET_DATASYNC_IDS=iC(lC),mC);var oC="tokens consistency service_params mss client_location entities adblock_detection response_received_commands store manifest player_preload shorts_prefetch".split(" "),pC=["type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.BrowseResponse","type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.PlayerResponse","type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.PanelResponse"];
function VB(a,b,c,d){this.u=a;this.fa=b;this.j=c;this.o=d;this.i=void 0;this.h=new Map;a.vc||(a.vc={});a.vc=Object.assign({},nC,a.vc)}
function qC(a,b,c){var d=rC;if(VB.instance!==void 0){if(c=VB.instance,a=[d!==c.u,a!==c.fa,b!==c.j,!1,!1,!1,void 0!==c.i],a.some(function(e){return e}))throw new U("InnerTubeTransportService is already initialized",a);
}else VB.instance=new VB(d,a,b,c)}
function ZB(a){var b={signalServiceEndpoint:{signal:"GET_DATASYNC_IDS"}};var c=c===void 0?tr:c;var d=sC(a,b);return d?new Gk(function(e,f){var g,h,k,l,m;return B(function(n){switch(n.h){case 1:return n.yield(d,2);case 2:g=n.i;h=g.u(b,void 0,c);if(!h){f(new U("Error: Failed to build request for command.",b));n.v(0);break}lA(h.input);l=((k=h.hb)==null?void 0:k.mode)==="cors"?"cors":void 0;if(a.j.ue){m=tC(h.config,l);n.v(4);break}return n.yield(uC(h.config,l),5);case 5:m=n.i;case 4:e(vC(a,h,m)),n.h=
0}})}):Lk(new U("Error: No request builder found for command.",b))}
function WB(a,b){function c(){}
var d="/youtubei/v1/"+TB(nz);var e=e===void 0?{Yb:{identity:tr}}:e;var f=f===void 0?!0:f;c=CB(BB(d));b.context||(b.context=gC(void 0,f));return new Gk(function(g){var h,k,l,m,n;return B(function(r){if(r.h==1)return h=Vy(d),k=xq(h)?"same-origin":"cors",a.j.ue?(l=tC(e,k),r.v(2)):r.yield(uC(e,k),3);r.h!=2&&(l=r.i);m=Wy(Vy(d));n={input:m,hb:Xy(m),Ja:b,config:e};g(vC(a,n,l,c));r.h=0})})}
function wC(a){var b,c,d,e,f,g;return B(function(h){switch(h.h){case 1:if(!((b=a)==null?0:(c=b.Ja)==null?0:c.context)){h.v(0);break}d=a.Ja.context;h.v(3);break;case 3:e=y([]),f=e.next();case 6:if(f.done){h.v(0);break}g=f.value;return h.yield(g.sj(d),7);case 7:f=e.next(),h.v(6)}})}
function xC(a,b,c){var d;if(b&&!(b==null?0:(d=b.sequenceMetaData)==null?0:d.skipProcessing)&&a.o){d=y(oC);for(var e=d.next();!e.done;e=d.next())e=e.value,a.o[e]&&a.o[e].handleResponse(b,c)}}
function vC(a,b,c,d){d=d===void 0?function(){}:d;
var e,f,g,h,k,l,m,n,r,t,w,z,x,H,F,V,na,ka,Od,Nb,hb,Sa,Ga,Ra,Dv,Ev,Fv,Gv;return B(function(oa){switch(oa.h){case 1:g=(e=b.config)==null?void 0:(f=e.Fj)==null?void 0:f.Ej;oa.v(2);break;case 3:h=oa.i;if(!h||h.isExpired()){oa.v(2);break}k=h.h();if(!T("web_process_response_store_responses")||h.isProcessed()){oa.v(5);break}xC(a,k,b);return oa.yield((void 0).mj(g),5);case 5:return oa.return(Promise.resolve(k));case 2:return oa.yield(wC(b),7);case 7:if((l=a.i)==null||!l.Aj(b.input,b.Ja)){oa.v(8);break}return oa.yield(a.i.jj(b.input,
b.Ja),9);case 9:return m=oa.i,xC(a,m,b),oa.return(m);case 8:return(t=(r=b.config)==null?void 0:r.requestKey)&&a.h.has(t)?n=a.h.get(t):(w=JSON.stringify(b.Ja),H=(x=(z=b.hb)==null?void 0:z.headers)!=null?x:{},b.hb=Object.assign({},b.hb,{headers:Object.assign({},H,c)}),F=Object.assign({},b.hb),b.hb.method==="POST"&&(F=Object.assign({},F,{body:w})),((V=b.config)==null?0:V.fg)&&zB(b.config.fg),na=function(){return a.fa.fetch(b.input,F,b.config)},n=na(),t&&a.h.set(t,n)),oa.yield(n,10);
case 10:(ka=oa.i)&&T("web_streaming_player")&&Array.isArray(ka)&&(ka=ka[0].playerResponse);if(ka&&"error"in ka&&((Od=ka)==null?0:(Nb=Od.error)==null?0:Nb.details))for(hb=ka.error.details,Sa=y(hb),Ga=Sa.next();!Ga.done;Ga=Sa.next())Ra=Ga.value,(Dv=Ra["@type"])&&pC.indexOf(Dv)>-1&&(delete Ra["@type"],ka=Ra);t&&a.h.has(t)&&a.h.delete(t);((Ev=b.config)==null?0:Ev.gg)&&zB(b.config.gg);if(ka||(Fv=a.i)==null||!Fv.Vi(b.input,b.Ja)){oa.v(11);break}return oa.yield(a.i.ij(b.input,b.Ja),12);case 12:ka=oa.i;case 11:return xC(a,
ka,b),((Gv=b.config)==null?0:Gv.cg)&&zB(b.config.cg),d(),oa.return(ka||void 0)}})}
function sC(a,b){a:{a=a.u;var c,d=(c=Ax(b,Np))==null?void 0:c.signal;if(d&&a.vc&&(c=a.vc[d])){var e=c();break a}var f;if((c=(f=Ax(b,Lp))==null?void 0:f.request)&&a.Xe&&(f=a.Xe[c])){e=f();break a}for(e in b)if(a.Qd[e]&&(b=a.Qd[e])){e=b();break a}e=void 0}if(e!==void 0)return Promise.resolve(e)}
function uC(a,b){var c,d,e,f;return B(function(g){if(g.h==1){e=(c=a)==null?void 0:(d=c.Yb)==null?void 0:d.sessionIndex;var h=g.yield;var k=sr({sessionIndex:e});if(!(k instanceof Gk)){var l=new Gk(Ek);Hk(l,2,k);k=l}return h.call(g,k,2)}f=g.i;return g.return(Promise.resolve(Object.assign({},hC(b),f)))})}
function tC(a,b){var c;a=a==null?void 0:(c=a.Yb)==null?void 0:c.sessionIndex;c=sr({sessionIndex:a});return Object.assign({},hC(b),c)}
;function yC(){}
v(yC,kC);yC.prototype.j=function(){return tz};
yC.prototype.i=function(a){return Ax(a,Yp)||void 0};
yC.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params);c.botguardResponse&&(a.botguardResponse=c.botguardResponse);c.feature&&(a.clientFeature=c.feature)};
da.Object.defineProperties(yC.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function zC(){}
v(zC,kC);zC.prototype.j=function(){return uz};
zC.prototype.i=function(a){return Ax(a,Xp)||void 0};
zC.prototype.h=function(a,b){b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params)};
da.Object.defineProperties(zC.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});var AC=new Vw("SHARE_CLIENT_PARAMS_PROVIDER_TOKEN");function BC(a){this.H=a}
v(BC,kC);BC.prototype.j=function(){return oz};
BC.prototype.i=function(a){return Ax(a,Rp)||Ax(a,Sp)||Ax(a,Qp)};
BC.prototype.h=function(a,b){b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);if(b.clientParamIdentifier){var c;if((c=this.H)==null?0:c.h(b.clientParamIdentifier))a.clientParams=this.H.i(b.clientParamIdentifier)}};
BC[Uw]=[AC];function CC(){}
v(CC,kC);CC.prototype.j=function(){return qz};
CC.prototype.i=function(a){return Ax(a,Pp)||void 0};
CC.prototype.h=function(a,b,c){a.feedbackTokens=[];b.feedbackToken&&a.feedbackTokens.push(b.feedbackToken);if(b=b.cpn||c.cpn)a.feedbackContext={cpn:b};a.isFeedbackTokenUnencrypted=!!c.is_feedback_token_unencrypted;a.shouldMerge=!1;c.extra_feedback_tokens&&(a.shouldMerge=!0,a.feedbackTokens=a.feedbackTokens.concat(c.extra_feedback_tokens))};
da.Object.defineProperties(CC.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function DC(){}
v(DC,kC);DC.prototype.j=function(){return qz};
DC.prototype.i=function(a){return Ax(a,Wp)};
DC.prototype.h=function(a,b){b.undoToken&&(a.feedbackTokens=[b.undoToken]);b.isUndoTokenUnencrypted&&(a.isFeedbackTokenUnencrypted=b.isUndoTokenUnencrypted)};
da.Object.defineProperties(DC.prototype,{o:{configurable:!0,enumerable:!0,get:function(){return!0}}});function EC(){}
v(EC,kC);EC.prototype.j=function(){return rz};
EC.prototype.i=function(a){return Ax(a,Vp)||void 0};
EC.prototype.h=function(a,b){b.params&&(a.params=b.params);b.secondaryParams&&(a.secondaryParams=b.secondaryParams)};function FC(){}
v(FC,kC);FC.prototype.j=function(){return sz};
FC.prototype.i=function(a){return Ax(a,Up)||void 0};
FC.prototype.h=function(a,b){b.actions&&(a.actions=b.actions);b.params&&(a.params=b.params);b.playlistId&&(a.playlistId=b.playlistId)};function GC(){}
v(GC,kC);GC.prototype.j=function(){return pz};
GC.prototype.i=function(a){return Ax(a,Tp)};
GC.prototype.h=function(a,b,c){c=c===void 0?{}:c;b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);c.includeListId&&(a.includeListId=!0)};function HC(){var a=a===void 0?uy:a;var b=b===void 0?{}:b;E("yt.logging.errors.log",ry);qy();ir(hr(),b);window.onerror=a;Xk=ty;window.addEventListener("unhandledrejection",function(c){if(c.reason instanceof Error){var d=c.reason;vy(d,{source:"unhandledrejection"});d.name==="AbortError"&&(d.level="WARNING")}ty(c.reason);c.preventDefault()})}
;function IC(){var a;return(a=S("WEB_PLAYER_CONTEXT_CONFIGS"))==null?void 0:a.WEB_PLAYER_CONTEXT_CONFIG_ID_EMBEDDED_PLAYER}
;var JC=D.caches,KC;function LC(a){var b=a.indexOf(":");return b===-1?{ee:a}:{ee:a.substring(0,b),datasyncId:a.substring(b+1)}}
function MC(){return B(function(a){if(KC!==void 0)return a.return(KC);KC=new Promise(function(b){var c;return B(function(d){switch(d.h){case 1:return wa(d,2),d.yield(JC.open("test-only"),4);case 4:return d.yield(JC.delete("test-only"),5);case 5:xa(d,3);break;case 2:if(c=ya(d),c instanceof Error&&c.name==="SecurityError")return b(!1),d.return();case 3:b("caches"in window),d.h=0}})});
return a.return(KC)})}
function NC(a){var b,c,d,e,f,g,h;B(function(k){if(k.h==1)return k.yield(MC(),2);if(k.h!=3){if(!k.i)return k.return(!1);b=[];return k.yield(JC.keys(),3)}c=k.i;d=y(c);for(e=d.next();!e.done;e=d.next())f=e.value,g=LC(f),h=g.datasyncId,!h||a.includes(h)||b.push(JC.delete(f));return k.return(Promise.all(b).then(function(l){return l.some(function(m){return m})}))})}
function OC(){var a,b,c,d,e,f,g;return B(function(h){if(h.h==1)return h.yield(MC(),2);if(h.h!=3){if(!h.i)return h.return(!1);a=Mr("cache contains other");return h.yield(JC.keys(),3)}b=h.i;c=y(b);for(d=c.next();!d.done;d=c.next())if(e=d.value,f=LC(e),(g=f.datasyncId)&&g!==a)return h.return(!0);return h.return(!1)})}
;function PC(){try{return!!self.sessionStorage}catch(a){return!1}}
;function QC(a){a=a.match(/(.*)::.*::.*/);if(a!==null)return a[1]}
function RC(a){if(PC()){var b=Object.keys(window.sessionStorage);b=y(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=QC(c);d===void 0||a.includes(d)||self.sessionStorage.removeItem(c)}}}
function SC(){if(!PC())return!1;var a=Mr(),b=Object.keys(window.sessionStorage);b=y(b);for(var c=b.next();!c.done;c=b.next())if(c=QC(c.value),c!==void 0&&c!==a)return!0;return!1}
;function TC(){YB().then(function(a){a&&(Wt(a),NC(a),gA(a),RC(a))})}
function UC(){var a=new fw;vn.ra(function(){var b,c,d,e,f;return B(function(g){switch(g.h){case 1:if(T("ytidb_clear_optimizations_killswitch")){g.v(2);break}b=Mr("clear");if(b.startsWith("V")&&b.endsWith("||")){var h=[b];Wt(h);NC(h);gA(h);RC(h);return g.return()}c=hA();d=SC();return g.yield(OC(),3);case 3:return e=g.i,g.yield(Xt(),4);case 4:if(f=g.i,!(c||d||e||f))return g.return();case 2:a.va()?TC():lk(a,"publicytnetworkstatus-online",TC),g.h=0}})})}
;function VC(a){return new Promise(function(b){window.setTimeout(b,a)})}
function WC(a,b,c){this.requestKey=a;this.o=b;this.i=c;this.u=function(){return new XMLHttpRequest};
this.h=void 0;this.j=[]}
WC.prototype.getLatestChallengeResponse=function(){return this.i};
function nm(a,b,c){var d,e,f,g;return B(function(h){if(h.h==1){wB();zB("att_fs",void 0,"attestation_challenge_fetch");if(!a.h)throw new el(9,"Missing fetcher");return h.yield(a.h(b,c),2)}d=h.i;f=(e=d)==null?void 0:e.bgChallenge;if(!f)throw new el(15,"Missing field");a.i=d;a.j.forEach(function(k){k(d)});
g=sA(f);zB("att_fc",void 0,"attestation_challenge_fetch");xB();return h.return(g)})}
function Xm(a,b){var c,d,e,f,g;return B(function(h){switch(h.h){case 1:c=new sj(100,3E5,.25,2),d=void 0;case 2:if(!(c.i<10)){h.v(4);break}wa(h,5);if(!(c.i>0)){h.v(7);break}return h.yield(VC(c.getValue()),7);case 7:return h.yield(XC(a,b),9);case 9:return e=h.i,h.return(e);case 5:f=ya(h);f instanceof el?d=f:(g=f instanceof Error?f.message:"Unknown",d=new el(9,g));tj(c);h.v(2);break;case 4:if(d)throw d;throw new el(9,"Unknown error");}})}
function XC(a,b){b=Yl(Zl(new Xl,b),a.requestKey);var c=new Ml,d=a.u();d.open("POST",a.o);d.setRequestHeader("X-Goog-Api-Key","AIzaSyDyT5W0Jh49F30Pqqtyfdf7pDLFKLJoAnw");d.setRequestHeader("Content-Type","application/json+protobuf");d.onload=function(){if(Cq(d)){var e=qn(d.responseText);c.resolve(e)}else c.reject(new el(cl(Dq(d)),d.statusText))};
d.onerror=function(){c.reject(new el(cl(Dq(d)),d.statusText))};
d.send(b.serialize());return c.promise}
function YC(a){var b={bicf:function(f){a.h=f},
blc:function(){return a.getLatestChallengeResponse()},
bcr:function(f){a.j.push(f)}},c=window;
c.ntpevasrs=b;if(c.ntpqfbel!==void 0)for(var d=y(c.ntpqfbel),e=d.next();!e.done;e=d.next())e=e.value,e(b);c.ntpqfbel=void 0}
;function ZC(a){var b,c;(b=a.ytcsi)==null||(c=b.tick)==null||c.call(b,"pot_ist")}
function $C(a){if(a instanceof Error){var b=G("yt.logging.errors.log");b&&b(a,"WARNING")}}
;function aD(a,b){var c=this;this.h=0;var d;this.Zb=(d=b==null?void 0:b.Zb)!=null?d:window;this.wd=b==null?void 0:b.wd;var e;this.requestKey=(e=b==null?void 0:b.requestKey)!=null?e:Kq("par_bir_key")||"O43z0dpjhgX20SCx4KAo";var f;this.Ce=(f=b==null?void 0:b.Ce)!=null?f:function(k){return new $l(k)};
var g;d=(g=b==null?void 0:b.cj)!=null?g:function(k,l,m){return new WC(k,l,m)};
this.bgChallenge=sA(a.bgChallenge);this.ttlSeconds=uA(tA(a.challenge||""));this.Oa=d(this.requestKey,T("par_at_ep")?["www.youtube.com","m.youtube.com"].includes(D.location.hostname)?"/api/jnn/v1/GenerateIT":"https://jnn-pa.googleapis.com/$rpc/google.internal.waa.v1.Waa/GenerateIT":"https://jnn-pa.googleapis.com/$rpc/google.internal.waa.v1.Waa/GenerateIT",a);this.Fe=b==null?void 0:b.Fe;YC(this.Oa);var h;this.le=(h=b==null?void 0:b.le)!=null?h:function(k){dk(c.Zb.document,"visibilitychange",function(){c.Zb.document.visibilityState===
"visible"&&k()})}}
function bD(a){if(!a.vm){var b={maxAttempts:5,ke:a.ttlSeconds*1E3};ZC(a.Zb);a.vm=a.Ce({Oa:a.Oa,Lb:{disable:T("html5_web_po_disable_remote_logging"),qa:"aGIf",nf:Jq(),Qf:T("wpo_dis_lfdms")?0:1E3,Rb:function(d){var e=dA.get(d);e||(e=new cA(d),e=new Zk(e),dA.set(d,e));return e}},
Sb:b,Af:a.bgChallenge,Pc:$C});a.h=Date.now();gm(a.vm,function(){a.h=Date.now()});
a.Zb.bgevmc={p:function(){var d;(d=a.vm)==null||d.pause()},
r:function(){var d;(d=a.vm)==null||d.resume()},
cr:function(){var d,e;return(e=(d=a.vm)==null?void 0:d.checkForRefresh())!=null?e:Promise.resolve()}};
Jc(a.vm,function(){return B(function(d){return d.return(cD(a))})});
var c=a.j.bind(a);a.wd&&a.ttlSeconds>0&&a.wd.then(function(d){d.listen("publicytnetworkstatus-online",c)});
a.le(c)}}
aD.prototype.j=function(){if(Date.now()>this.h+this.ttlSeconds*1E3){var a;(a=this.vm)==null||em(a)}};
function cD(a){if(a.i)return a.i;if(!a.vm)throw Error("VMNI");a.i=new Tm({vm:a.vm,Oa:a.Oa,Jd:!0,onError:$C,Sb:a.Fe});return a.i}
function dD(a,b){a=new aD(a,b);bD(a);(b==null?0:b.Yi)||cD(a)}
function eD(a){try{var b=JSON.parse(a);if(b.bgChallenge)return b}catch(c){}}
function fD(){var a=window,b={};a=a===void 0?window:a;var c=a.ytAtR,d;b==null||(d=b.Wd)==null||d.tj();if(c){if(c=eD(c)){var e;b==null||(e=b.Wd)==null||e.je("SUCCESS");dD(c,b)}a.ytAtR=void 0}else a.ytAtRC=function(f){if(f=eD(f)){var g;b==null||(g=b.Wd)==null||g.je("SUCCESS");dD(f,b);a.ytAtRC=void 0}}}
;var gD=["www.youtube-nocookie.com","www.youtubeeducation.com","youtube.googleapis.com"];function hD(){this.state=1;this.vm=null;this.h=void 0}
p=hD.prototype;p.initialize=function(a,b,c,d){this.h=d;if(a.program){var e;d=(e=a.interpreterUrl)!=null?e:null;if(a.interpreterSafeScript)e=Dp(a.interpreterSafeScript);else{var f;e=(f=a.interpreterScript)!=null?f:null}a.interpreterSafeUrl&&(d=Ep(a.interpreterSafeUrl).toString());iD(this,e,d,a.program,b,c)}else sy(Error("BL:CIP"))};
function iD(a,b,c,d,e,f){var g=g===void 0?"trayride":g;c?(a.state=2,xz(Fp(c),function(){window[g]?jD(a,d,g,e):(a.state=3,zz(c),sy(new U("BL:ULB",""+c)))},f)):b?(f=Ti("SCRIPT"),b instanceof Hb?(f.textContent=Jb(b),Kb(f)):f.textContent=b,f.nonce=Gb(document),document.head.appendChild(f),document.head.removeChild(f),window[g]?jD(a,d,g,e):(a.state=4,sy(new U("BL:ULBJ")))):sy(new U("BL:ULV"))}
p.isLoading=function(){return this.state===2};
function jD(a,b,c,d){a.state=5;var e=!!a.h&&gD.includes(lc(a.h)||"");try{var f=new Ol({program:b,globalName:c,Lb:{disable:!T("att_web_record_metrics")||!T("att_skip_metrics_for_cookieless_domains_ks")&&e,qa:"aGIf"}});f.Pb.then(function(){a.state=6;d&&d(b)});
a.Cd(f)}catch(g){a.state=7,g instanceof Error&&sy(g)}}
p.invoke=function(a){a=a===void 0?{}:a;return this.Hd()?this.De({Ha:a}):null};
p.dispose=function(){this.Cd(null);this.state=8};
p.Hd=function(){return!!this.vm};
p.De=function(a){return this.vm.se(a)};
p.Cd=function(a){yc(this.vm);this.vm=a};function kD(){var a=G("yt.abuse.playerAttLoader");return a&&["bgvma","bgvmb","bgvmc"].every(function(b){return b in a})?a:null}
;function lD(){hD.apply(this,arguments)}
v(lD,hD);lD.prototype.Cd=function(a){var b;(b=kD())==null||b.bgvma();a?(b={bgvma:a.dispose.bind(a),bgvmb:a.snapshot.bind(a),bgvmc:a.se.bind(a)},E("yt.abuse.playerAttLoader",b),E("yt.abuse.playerAttLoaderRun",function(c){return a.snapshot(c)})):(E("yt.abuse.playerAttLoader",null),E("yt.abuse.playerAttLoaderRun",null))};
lD.prototype.Hd=function(){return!!kD()};
lD.prototype.De=function(a){return kD().bgvmc(a)};var mD=new Vw("AUTH_SERVICE_TOKEN");function nD(a){lx.call(this,a===void 0?"document_active":a);var b=this;this.o=10;this.h=new Map;this.transitions=[{from:"document_active",to:"document_disposed_preventable",action:this.G},{from:"document_active",to:"document_disposed",action:this.u},{from:"document_disposed_preventable",to:"document_disposed",action:this.u},{from:"document_disposed_preventable",to:"flush_logs",action:this.H},{from:"document_disposed_preventable",to:"document_active",action:this.i},{from:"document_disposed",to:"flush_logs",
action:this.H},{from:"document_disposed",to:"document_active",action:this.i},{from:"document_disposed",to:"document_disposed",action:function(){}},
{from:"flush_logs",to:"document_active",action:this.i}];window.addEventListener("pagehide",function(c){b.transition("document_disposed",{event:c})});
window.addEventListener("beforeunload",function(c){b.transition("document_disposed_preventable",{event:c})})}
v(nD,lx);nD.prototype.G=function(a,b){if(!this.h.get("document_disposed_preventable")){a(b==null?void 0:b.event);var c,d;if((b==null?0:(c=b.event)==null?0:c.defaultPrevented)||(b==null?0:(d=b.event)==null?0:d.returnValue)){b.event.returnValue||(b.event.returnValue=!0);b.event.defaultPrevented||b.event.preventDefault();this.h=new Map;this.transition("document_active");return}}this.h.set("document_disposed_preventable",!0);this.h.get("document_disposed")?this.transition("flush_logs"):this.transition("document_disposed")};
nD.prototype.u=function(a,b){this.h.get("document_disposed")?this.transition("document_active"):(a(b==null?void 0:b.event),this.h.set("document_disposed",!0),this.transition("flush_logs"))};
nD.prototype.H=function(a,b){a(b==null?void 0:b.event);this.transition("document_active")};
nD.prototype.i=function(){this.h=new Map};function oD(a){lx.call(this,a===void 0?"document_visibility_unknown":a);var b=this;this.transitions=[{from:"document_visibility_unknown",to:"document_visible",action:this.i},{from:"document_visibility_unknown",to:"document_hidden",action:this.h},{from:"document_visibility_unknown",to:"document_foregrounded",action:this.H},{from:"document_visibility_unknown",to:"document_backgrounded",action:this.u},{from:"document_visible",to:"document_hidden",action:this.h},{from:"document_visible",to:"document_foregrounded",
action:this.H},{from:"document_visible",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_visible",action:this.i},{from:"document_foregrounded",to:"document_hidden",action:this.h},{from:"document_foregrounded",to:"document_foregrounded",action:this.H},{from:"document_hidden",to:"document_visible",action:this.i},{from:"document_hidden",to:"document_backgrounded",action:this.u},{from:"document_hidden",to:"document_hidden",action:this.h},{from:"document_backgrounded",to:"document_hidden",
action:this.h},{from:"document_backgrounded",to:"document_backgrounded",action:this.u},{from:"document_backgrounded",to:"document_visible",action:this.i}];document.addEventListener("visibilitychange",function(c){document.visibilityState==="visible"?b.transition("document_visible",{event:c}):b.transition("document_hidden",{event:c})});
T("visibility_lifecycles_dynamic_backgrounding")&&(window.addEventListener("blur",function(c){b.transition("document_backgrounded",{event:c})}),window.addEventListener("focus",function(c){b.transition("document_foregrounded",{event:c})}))}
v(oD,lx);oD.prototype.i=function(a,b){a(b==null?void 0:b.event);T("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_foregrounded")};
oD.prototype.h=function(a,b){a(b==null?void 0:b.event);T("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_backgrounded")};
oD.prototype.u=function(a,b){a(b==null?void 0:b.event)};
oD.prototype.H=function(a,b){a(b==null?void 0:b.event)};function pD(){this.o=new nD;this.u=new oD}
pD.prototype.install=function(){var a=C.apply(0,arguments),b=this;a.forEach(function(c){b.o.install(c)});
a.forEach(function(c){b.u.install(c)})};function qD(){this.o=[];this.i=new Map;this.h=new Map;this.j=new Set}
qD.prototype.clickCommand=function(a,b,c){var d=a.clickTrackingParams;c=c===void 0?0:c;if(d)if(c=Ly(c===void 0?0:c)){a=this.client;d=new Ey({trackingParams:d});var e=void 0;if(T("no_client_ve_attach_unless_shown")){var f=Zz(d,c);Vz.set(f,!0);$z(d,c)}e=e||"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK";f=Yz({cttAuthInfo:Ny(c)||void 0,automatedLogEventSource:void 0},c);d={csn:c,ve:d.getAsJson(),gestureType:e};b&&(d.clientData=b);c==="UNDEFINED_CSN"?aA("visualElementGestured",f,d):a?iy("visualElementGestured",
d,a,f):Es("visualElementGestured",d,f);b=!0}else b=!1;else b=!1;return b};
qD.prototype.stateChanged=function(a,b,c){this.visualElementStateChanged(new Ey({trackingParams:a}),b,c===void 0?0:c)};
qD.prototype.visualElementStateChanged=function(a,b,c){c=c===void 0?0:c;if(c===0&&this.j.has(c))this.o.push([a,b]);else{var d=c;d=d===void 0?0:d;c=Ly(d);a||(a=(a=Iy(d===void 0?0:d))?new Ey({veType:a,youtubeData:void 0,jspbYoutubeData:void 0}):null);var e=a;c&&e&&(a=this.client,d=Yz({cttAuthInfo:Ny(c)||void 0},c),b={csn:c,ve:e.getAsJson(),clientData:b},c==="UNDEFINED_CSN"?aA("visualElementStateChanged",d,b):a?iy("visualElementStateChanged",b,a,d):Es("visualElementStateChanged",b,d))}};
function rD(a,b){if(b===void 0)for(var c=Ky(),d=0;d<c.length;d++)c[d]!==void 0&&rD(a,c[d]);else a.i.forEach(function(e,f){(f=a.h.get(f))&&Xz(a.client,b,f,e)}),a.i.clear(),a.h.clear()}
;function sD(){pD.call(this);var a={};this.install((a.document_disposed={callback:this.h},a));T("combine_ve_grafts")&&(a={},this.install((a.document_disposed={callback:this.i},a)));a={};this.install((a.flush_logs={callback:this.j},a));T("web_log_cfg_cee_ks")||Or(tD)}
v(sD,pD);sD.prototype.j=function(){Es("finalPayload",{csn:Ly()})};
sD.prototype.h=function(){yy(zy)};
sD.prototype.i=function(){var a=rD;qD.instance||(qD.instance=new qD);a(qD.instance)};
function tD(){var a=S("CLIENT_EXPERIMENT_EVENTS");if(a){var b=Ce();a=y(a);for(var c=a.next();!c.done;c=a.next())c=c.value,b(c)&&Es("genericClientExperimentEvent",{eventType:c});delete fq.CLIENT_EXPERIMENT_EVENTS}}
;function uD(a,b){var c=C.apply(2,arguments);a=a===void 0?0:a;U.call(this,b,c);this.errorType=a;Object.setPrototypeOf(this,this.constructor.prototype)}
v(uD,U);var vD=new Vw("NETWORK_SLI_TOKEN");function wD(a){this.h=a}
wD.prototype.fetch=function(a,b,c){var d=this,e;return B(function(f){e=xD(d,a,b);return f.return(fetch(e).then(function(g){return d.handleResponse(g,c)}).catch(function(g){sy(g);
if((c==null?0:c.mf)&&g instanceof uD&&g.errorType===1)return Promise.reject(g)}))})};
function xD(a,b,c){if(a.h){var d=mc(wc(b,"key"))||"/UNKNOWN_PATH";a.h.start(d)}a=c;T("wug_networking_gzip_request")&&(a=jv(c));return new window.Request(b,a)}
wD.prototype.handleResponse=function(a,b){var c=a.text().then(function(d){if((b==null?0:b.Jf)&&a.ok)return Yg(b.Jf,d);d=d.replace(")]}'","");if((b==null?0:b.mf)&&d)try{var e=JSON.parse(d)}catch(g){throw new uD(1,"JSON parsing failed after fetch");}var f;return(f=e)!=null?f:JSON.parse(d)});
a.redirected||a.ok?this.h&&this.h.success():(this.h&&this.h.bj(),c=c.then(function(d){sy(new U("Error: API fetch failed",a.status,a.url,d));return Object.assign({},d,{errorMetadata:{status:a.status}})}));
return c};
wD[Uw]=[new Ww(vD)];var yD=new Vw("NETWORK_MANAGER_TOKEN");function zD(){}
function AD(){var a=G("ytglobal.storage_");a||(a=new zD,E("ytglobal.storage_",a));return a}
zD.prototype.estimate=function(){var a,b,c;return B(function(d){a=navigator;return((b=a.storage)==null?0:b.estimate)?d.return(a.storage.estimate()):((c=a.webkitTemporaryStorage)==null?0:c.queryUsageAndQuota)?d.return(BD()):d.return()})};
function BD(){var a=navigator;return new Promise(function(b,c){var d;(d=a.webkitTemporaryStorage)!=null&&d.queryUsageAndQuota?a.webkitTemporaryStorage.queryUsageAndQuota(function(e,f){b({usage:e,quota:f})},function(e){c(e)}):c(Error("webkitTemporaryStorage is not supported."))})}
E("ytglobal.storageClass_",zD);function Cs(a,b){var c=this;this.handleError=a;this.h=b;this.i=!1;self.document===void 0||self.addEventListener("beforeunload",function(){c.i=!0});
this.j=Math.random()<=.2}
Cs.prototype.wa=function(a){this.handleError(a)};
Cs.prototype.logEvent=function(a,b){switch(a){case "IDB_DATA_CORRUPTED":T("idb_data_corrupted_killswitch")||this.h("idbDataCorrupted",b);break;case "IDB_UNEXPECTEDLY_CLOSED":this.h("idbUnexpectedlyClosed",b);break;case "IS_SUPPORTED_COMPLETED":T("idb_is_supported_completed_killswitch")||this.h("idbIsSupportedCompleted",b);break;case "QUOTA_EXCEEDED":CD(this,b);break;case "TRANSACTION_ENDED":this.j&&Math.random()<=.1&&this.h("idbTransactionEnded",b);break;case "TRANSACTION_UNEXPECTEDLY_ABORTED":a=
Object.assign({},b,{hasWindowUnloaded:this.i}),this.h("idbTransactionAborted",a)}};
function CD(a,b){AD().estimate().then(function(c){c=Object.assign({},b,{isSw:self.document===void 0,isIframe:self!==self.top,deviceStorageUsageMbytes:DD(c==null?void 0:c.usage),deviceStorageQuotaMbytes:DD(c==null?void 0:c.quota)});a.h("idbQuotaExceeded",c)})}
function DD(a){return typeof a==="undefined"?"-1":String(Math.ceil(a/1048576))}
;var rC={Qd:{feedbackEndpoint:iC(CC),modifyChannelNotificationPreferenceEndpoint:iC(EC),playlistEditEndpoint:iC(FC),shareEntityEndpoint:iC(BC),subscribeEndpoint:iC(yC),undoFeedbackEndpoint:iC(DC),unsubscribeEndpoint:iC(zC),webPlayerShareEntityServiceEndpoint:iC(GC)}};function ED(){var a=bx();Yw(a,{oc:yD,Gd:wD});Yw(a,{oc:mD,Gd:rr});var b=fC(),c=a.resolve(mD),d=a.resolve(yD),e={};b&&(e.client_location=b);qC(d,c,e);Yw(a,{oc:XB,ze:VB.instance})}
;var FD={},GD=(FD["api.invalidparam"]=2,FD.auth=150,FD["drm.auth"]=150,FD["heartbeat.net"]=150,FD["heartbeat.servererror"]=150,FD["heartbeat.stop"]=150,FD["html5.unsupportedads"]=5,FD["fmt.noneavailable"]=5,FD["fmt.decode"]=5,FD["fmt.unplayable"]=5,FD["html5.missingapi"]=5,FD["html5.unsupportedlive"]=5,FD["drm.unavailable"]=5,FD["mrm.blocked"]=151,FD["embedder.identity.denied"]=152,FD["embedder.identity.missing.referrer"]=153,FD);var HD=new Set("endSeconds startSeconds mediaContentUrl suggestedQuality videoId rct rctn playmuted muted_autoplay_duration_mode".split(" "));function ID(a){return(a.search("cue")===0||a.search("load")===0)&&a!=="loadModule"}
function JD(a,b,c){if(typeof a==="string")return{videoId:a,startSeconds:b,suggestedQuality:c};b={};c=y(HD);for(var d=c.next();!d.done;d=c.next())d=d.value,a[d]&&(b[d]=a[d]);if(a=a.embedConfig||a.embed_config)if(typeof a==="string")b.embed_config=a;else if(Pa(a))try{var e=JSON.stringify(a);b.embed_config=e}catch(f){console.error("Invalid embedConfig JSON",f)}return b}
function KD(a,b,c,d){if(Pa(a)&&!Array.isArray(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};typeof a==="string"&&a.length===16?b.list="PL"+a:b.playlist=a;return b}
;function LD(a,b){J.call(this);var c=this;this.api=a;this.j=this.G=!1;this.K=[];this.V={};this.o=[];this.i=[];this.ba=!1;this.sessionId=this.h=null;this.targetOrigin="*";this.aa=T("web_player_split_event_bus_iframe");this.B=S("POST_MESSAGE_ORIGIN")||document.location.protocol+"//"+document.location.hostname;this.u=function(d){c.onMessage(d)};
MD.addEventListener("message",this.u);if(a=S("WIDGET_ID"))this.sessionId=a;b&&this.u(b);ND(this,"onReady",function(){c.G=!0;var d=c.api.getVideoData();if(!d.isPlayable){c.ba=!0;d=d.errorCode;var e=e===void 0?5:e;c.errorCode=d?GD[d]||e:e;c.sendMessage("onError",Number(c.errorCode))}OD(c);c.h||c.j||window.parent===window||!c.sessionId||PD(c,{event:"readyToListen"},window.parent)});
ND(this,"onVideoProgress",this.qg.bind(this));ND(this,"onVolumeChange",this.rg.bind(this));ND(this,"onApiChange",this.jg.bind(this));ND(this,"onPlaybackQualityChange",this.ng.bind(this));ND(this,"onPlaybackRateChange",this.og.bind(this));ND(this,"onStateChange",this.pg.bind(this));ND(this,"onWebglSettingsChanged",this.sg.bind(this));ND(this,"onCaptionsTrackListChanged",this.kg.bind(this));ND(this,"captionssettingschanged",this.lg.bind(this))}
v(LD,J);function OD(a){if(a.h)if(a.j)a.sendMessage("alreadyInitialized");else if(a.G){a.j=!0;a.G=!1;a.sendMessage("initialDelivery",QD(a));a.sendMessage("onReady");zB("ep_init_ar");for(var b=y(a.K),c=b.next();!c.done;c=b.next())PD(a,c.value);a.K=[]}}
function RD(a,b){a.sendMessage("infoDelivery",b)}
p=LD.prototype;p.sendMessage=function(a,b){a={event:a,info:b===void 0?null:b};this.j?PD(this,a):this.K.push(a)};
function SD(a,b,c){return function(d){b==="onError"?a.api.logApiCall(b+" invocation",c,d):a.api.logApiCall(b+" invocation",c);a.sendMessage(b,d)}}
function ND(a,b,c){a.o.push({eventType:b,listener:c});a.api.addEventListener(b,c)}
function QD(a){if(!a.api)return null;var b=a.api.getApiInterface();Yb(b,"getVideoData");for(var c={apiInterface:b},d=0,e=b.length;d<e;d++){var f=b[d];if(f.search("get")===0||f.search("is")===0){var g=0;f.search("get")===0?g=3:f.search("is")===0&&(g=2);g=f.charAt(g).toLowerCase()+f.substring(g+1);try{var h=a.api[f]();c[g]=h}catch(k){}}}c.videoData=a.api.getVideoData();c.currentTimeLastUpdated_=Date.now()/1E3;return c}
p.pg=function(a){a={playerState:a,currentTime:this.api.getCurrentTime(),duration:this.api.getDuration(),videoData:this.api.getVideoData(),videoStartBytes:0,videoBytesTotal:this.api.getVideoBytesTotal(),videoLoadedFraction:this.api.getVideoLoadedFraction(),playbackQuality:this.api.getPlaybackQuality(),availableQualityLevels:this.api.getAvailableQualityLevels(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getVideoUrl&&
(a.videoUrl=this.api.getVideoUrl());this.api.getVideoContentRect&&(a.videoContentRect=this.api.getVideoContentRect());this.api.getProgressState&&(a.progressState=this.api.getProgressState());this.api.getPlaylist&&(a.playlist=this.api.getPlaylist());this.api.getPlaylistIndex&&(a.playlistIndex=this.api.getPlaylistIndex());RD(this,a)};
p.ng=function(a){a={playbackQuality:a};this.api.getAvailableQualityLevels&&(a.availableQualityLevels=this.api.getAvailableQualityLevels());this.api.getPreferredQuality&&(a.preferredQuality=this.api.getPreferredQuality());RD(this,a)};
p.og=function(a){RD(this,{playbackRate:a})};
p.jg=function(){for(var a=this.api.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.api.getOptions(e);a.join(", ");b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var k=f[g],l=this.api.getOption(e,k);b[e][k]=l}}this.sendMessage("apiInfoDelivery",b)};
p.rg=function(){RD(this,{muted:this.api.isMuted(),volume:this.api.getVolume()})};
p.qg=function(a){a={currentTime:a,videoBytesLoaded:this.api.getVideoBytesLoaded(),videoLoadedFraction:this.api.getVideoLoadedFraction(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getProgressState&&(a.progressState=this.api.getProgressState());RD(this,a)};
p.sg=function(){RD(this,{sphericalProperties:this.api.getSphericalProperties()})};
p.kg=function(){if(this.api.getCaptionTracks){var a={captionTracks:this.api.getCaptionTracks()};RD(this,a)}};
p.lg=function(){if(this.api.getSubtitlesUserSettings){var a={subtitlesUserSettings:this.api.getSubtitlesUserSettings()};RD(this,a)}};
p.onMessage=function(a){if(!(this.B!=="*"&&a.origin!==this.B||this.h&&a.source!==this.h||typeof a.data!=="string")){try{var b=JSON.parse(a.data)}catch(f){return}if(b)switch(b.event){case "listening":var c=a.source;a=a.origin;b=b.id;a!=="null"&&(this.B=this.targetOrigin=a);this.h=c;this.sessionId=b;OD(this);break;case "command":if(c=b.func,b=b.args,c==="addEventListener"&&b)c=b[0],b=a.origin,c==="onReady"?this.api.logApiCall(c+" invocation",b):c==="onError"&&this.ba&&(this.api.logApiCall(c+" invocation",
b,this.errorCode),this.errorCode=void 0),this.api.logApiCall(c+" registration",b),this.V[c]||c==="onReady"||(a=SD(this,c,b),this.i.push({eventType:c,listener:a,origin:b}),this.aa?this.api.handleExternalCall("addEventListener",[c,a],b):this.api.addEventListener(c,a),this.V[c]=!0);else if(a=a.origin,this.api.isExternalMethodAvailable(c,a)){b=b||[];if(b.length>0&&ID(c)){var d=b;if(Pa(d[0])&&!Array.isArray(d[0]))var e=d[0];else switch(e={},c){case "loadVideoById":case "cueVideoById":e=JD(d[0],d[1]!==
void 0?Number(d[1]):void 0,d[2]);break;case "loadVideoByUrl":case "cueVideoByUrl":e=d[0];typeof e==="string"&&(e={mediaContentUrl:e,startSeconds:d[1]!==void 0?Number(d[1]):void 0,suggestedQuality:d[2]});b:{if((d=e.mediaContentUrl)&&(d=/\/([ve]|embed)\/([^#?]+)/.exec(d))&&d[2]){d=d[2];break b}d=null}e.videoId=d;e=JD(e);break;case "loadPlaylist":case "cuePlaylist":e=KD(d[0],d[1],d[2],d[3])}b.length=1;b[0]=e}this.api.handleExternalCall(c,b,a);ID(c)&&RD(this,QD(this))}}}};
function PD(a,b,c){if(c=c===void 0?a.h:c){b.channel="widget";a.sessionId&&(b.id=a.sessionId);try{var d=JSON.stringify(b);c.postMessage(d,a.targetOrigin)}catch(e){sy(e)}}}
p.X=function(){J.prototype.X.call(this);MD.removeEventListener("message",this.u);for(var a=0;a<this.o.length;a++){var b=this.o[a];this.api.removeEventListener(b.eventType,b.listener)}this.o=[];for(a=0;a<this.i.length;a++)b=this.i[a],this.aa?this.api.handleExternalCall("removeEventListener",[b.eventType,b.listener],b.origin):this.api.removeEventListener(b.eventType,b.listener);this.i=[]};
var MD=window;function TD(a,b,c){J.call(this);var d=this;this.api=a;this.id=b;this.origin=c;this.h={};this.j=T("web_player_split_event_bus_iframe");this.i=function(e){d.onMessage(e)};
UD.addEventListener("message",this.i);VD(this,"RECEIVING")}
v(TD,J);p=TD.prototype;p.addListener=function(a,b){if(!(a in this.h)){var c=this.mg.bind(this,a);this.h[a]=c;this.addEventListener(a,c,b)}};
p.mg=function(a,b){this.I||VD(this,a,WD(a,b))};
p.removeListener=function(a,b){a in this.h&&(this.removeEventListener(a,this.h[a],b),delete this.h[a])};
p.addEventListener=function(a,b,c){this.j?a==="onReady"?this.api.addEventListener(a,b):this.api.handleExternalCall("addEventListener",[a,b],c||null):this.api.addEventListener(a,b)};
p.removeEventListener=function(a,b,c){this.j?a==="onReady"?this.api.removeEventListener(a,b):this.api.handleExternalCall("removeEventListener",[a,b],c||null):this.api.removeEventListener(a,b)};
function XD(a,b){switch(a){case "loadVideoById":return[JD(b)];case "cueVideoById":return[JD(b)];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return[KD(b)];case "cuePlaylist":return[KD(b)];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];case "setShuffle":return[b.shufflePlaylist];
case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey,b.key,b.code]}return[]}
function YD(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
function WD(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}if(b!=null)return{value:b}}
function VD(a,b,c){a.I||(b={id:a.id,command:b},c&&(b.data=c),ZD.postMessage(JSON.stringify(b),a.origin))}
p.onMessage=function(a){if(a.origin===this.origin){var b=a.data;if(typeof b==="string"){try{b=JSON.parse(b)}catch(e){return}if(b.command){var c=b.command;b=b.data;a=a.origin;if(!this.I){var d=b||{};switch(c){case "addEventListener":typeof d.event==="string"&&this.addListener(d.event,a);break;case "removeEventListener":typeof d.event==="string"&&this.removeListener(d.event,a);break;default:this.api.isReady()&&this.api.isExternalMethodAvailable(c,a||null)&&(b=XD(c,b||{}),b=this.api.handleExternalCall(c,
b,a||null),(b=YD(c,b))&&VD(this,c,b))}}}}}};
p.X=function(){UD.removeEventListener("message",this.i);for(var a in this.h)this.h.hasOwnProperty(a)&&this.removeListener(a);J.prototype.X.call(this)};
var UD=window,ZD=window.parent;var $D=new lD;function aE(){return $D.Hd()}
function bE(a){a=a===void 0?{}:a;return $D.invoke(a)}
;function cE(a,b,c,d,e){J.call(this);var f=this;this.B=b;this.webPlayerContextConfig=d;this.Ub=e;this.Va=!1;this.api={};this.na=this.u=null;this.V=new O;this.h={};this.ba=this.za=this.elementId=this.Wa=this.config=null;this.aa=!1;this.j=this.G=null;this.Ga={};this.gd=["onReady"];this.lastError=null;this.lb=NaN;this.K={};this.ha=0;this.i=this.o=a;Ac(this,this.V);dE(this);c?this.ha=setTimeout(function(){f.loadNewVideoConfig(c)},0):d&&(eE(this),fE(this))}
v(cE,J);p=cE.prototype;p.getId=function(){return this.B};
p.loadNewVideoConfig=function(a){if(!this.I){this.ha&&(clearTimeout(this.ha),this.ha=0);var b=a||{};b instanceof mz||(b=new mz(b));this.config=b;this.setConfig(a);fE(this);this.isReady()&&gE(this)}};
function eE(a){var b;a.webPlayerContextConfig?b=a.webPlayerContextConfig.rootElementId:b=a.config.attrs.id;a.elementId=b||a.elementId;a.elementId==="video-player"&&(a.elementId=a.B,a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.B:a.config.attrs.id=a.B);var c;((c=a.i)==null?void 0:c.id)===a.elementId&&(a.elementId+="-player",a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.elementId:a.config.attrs.id=a.elementId)}
p.setConfig=function(a){this.Wa=a;this.config=hE(a);eE(this);if(!this.za){var b;this.za=iE(this,((b=this.config.args)==null?void 0:b.jsapicallback)||"onYouTubePlayerReady")}this.config.args?this.config.args.jsapicallback=null:this.config.args={jsapicallback:null};var c;if((c=this.config)==null?0:c.attrs)a=this.config.attrs,(b=a.width)&&this.i&&(this.i.style.width=nn(Number(b)||b)),(a=a.height)&&this.i&&(this.i.style.height=nn(Number(a)||a))};
function gE(a){if(a.config&&a.config.loaded!==!0)if(a.config.loaded=!0,!a.config.args||a.config.args.autoplay!=="0"&&a.config.args.autoplay!==0&&a.config.args.autoplay!==!1){var b;a.api.loadVideoByPlayerVars((b=a.config.args)!=null?b:null)}else a.api.cueVideoByPlayerVars(a.config.args)}
function jE(a){var b=!0,c=kE(a);c&&a.config&&(b=c.dataset.version===lE(a));return b&&!!G("yt.player.Application.create")}
function fE(a){if(!a.I&&!a.aa){var b=jE(a);if(b&&(kE(a)?"html5":null)==="html5")a.ba="html5",a.isReady()||mE(a);else if(nE(a),a.ba="html5",b&&a.j&&a.o)a.o.appendChild(a.j),mE(a);else{a.config&&(a.config.loaded=!0);var c=!1;a.G=function(){c=!0;var d=oE(a,"player_bootstrap_method")?G("yt.player.Application.createAlternate")||G("yt.player.Application.create"):G("yt.player.Application.create");var e=a.config?hE(a.config):void 0;d&&d(a.o,e,a.webPlayerContextConfig,a.Ub);mE(a)};
a.aa=!0;b?a.G():(xz(lE(a),a.G),(b=pE(a))&&Ez(b||""),qE(a)&&!c&&E("yt.player.Application.create",null))}}}
function kE(a){var b=Si(a.elementId);!b&&a.i&&a.i.querySelector&&(b=a.i.querySelector("#"+a.elementId));return b}
function mE(a){if(!a.I){var b=kE(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);if(c){a.aa=!1;if(!oE(a,"html5_remove_not_servable_check_killswitch")){var d;if((b==null?0:b.isNotServable)&&a.config&&(b==null?0:b.isNotServable((d=a.config.args)==null?void 0:d.video_id)))return}rE(a)}else a.lb=setTimeout(function(){mE(a)},50)}}
function rE(a){dE(a);a.Va=!0;var b=kE(a);if(b){a.u=sE(a,b,"addEventListener");a.na=sE(a,b,"removeEventListener");var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=a.api,e=0;e<c.length;e++){var f=c[e];d[f]||(d[f]=sE(a,b,f))}}for(var g in a.h)a.h.hasOwnProperty(g)&&a.u&&a.u(g,a.h[g]);gE(a);a.za&&a.za(a.api);a.V.yb("onReady",a.api)}
function sE(a,b,c){var d=b[c];return function(){var e=C.apply(0,arguments);try{return a.lastError=null,d.apply(b,e)}catch(f){if(c!=="sendAbandonmentPing")throw f.params=c,a.lastError=f,e=new U("PlayerProxy error in method call",{error:f,method:c,playerId:a.B}),e.level="WARNING",e;}}}
function dE(a){a.Va=!1;if(a.na)for(var b in a.h)a.h.hasOwnProperty(b)&&a.na(b,a.h[b]);for(var c in a.K)a.K.hasOwnProperty(c)&&clearTimeout(Number(c));a.K={};a.u=null;a.na=null;b=a.api;for(var d in b)b.hasOwnProperty(d)&&(b[d]=null);b.addEventListener=function(e,f){a.addEventListener(e,f)};
b.removeEventListener=function(e,f){a.removeEventListener(e,f)};
b.destroy=function(){a.dispose()};
b.getLastError=function(){return a.getLastError()};
b.getPlayerType=function(){return a.getPlayerType()};
b.getCurrentVideoConfig=function(){return a.Wa};
b.loadNewVideoConfig=function(e){a.loadNewVideoConfig(e)};
b.isReady=function(){return a.isReady()}}
p.isReady=function(){return this.Va};
p.addEventListener=function(a,b){var c=this,d=iE(this,b);d&&(Sb(this.gd,a)>=0||this.h[a]||(b=tE(this,a),this.u&&this.u(a,b)),this.V.subscribe(a,d),a==="onReady"&&this.isReady()&&setTimeout(function(){d(c.api)},0))};
p.removeEventListener=function(a,b){this.I||(b=iE(this,b))&&this.V.unsubscribe(a,b)};
function iE(a,b){var c=b;if(typeof b==="string"){if(a.Ga[b])return a.Ga[b];c=function(){var d=C.apply(0,arguments),e=G(b);if(e)try{e.apply(D,d)}catch(f){throw d=new U("PlayerProxy error when executing callback",{error:f}),d.level="ERROR",d;}};
a.Ga[b]=c}return c?c:null}
function tE(a,b){function c(d){function e(){if(!a.I)try{a.V.yb(b,d!=null?d:void 0)}catch(h){var g=new U("PlayerProxy error when creating global callback",{error:h.message,event:b,playerId:a.B,data:d,originalStack:h.stack,componentStack:h.We});g.level="WARNING";throw g;}}
if(oE(a,"web_player_publish_events_immediately"))e();else{var f=setTimeout(function(){e();var g=a.K,h=String(f);h in g&&delete g[h]},0);
Ji(a.K,String(f))}}
return a.h[b]=c}
p.getPlayerType=function(){return this.ba||(kE(this)?"html5":null)};
p.getLastError=function(){return this.lastError};
function nE(a){a.cancel();dE(a);a.ba=null;a.config&&(a.config.loaded=!1);var b=kE(a);b&&(jE(a)||!qE(a)?a.j=b:(b&&b.destroy&&b.destroy(),a.j=null));a.o&&Ui(a.o)}
p.cancel=function(){this.G&&Bz(lE(this),this.G);clearTimeout(this.lb);this.aa=!1};
p.X=function(){nE(this);if(this.j&&this.config&&this.j.destroy)try{this.j.destroy()}catch(b){var a=new U("PlayerProxy error during disposal",{error:b});a.level="ERROR";throw a;}this.Ga=null;for(a in this.h)this.h.hasOwnProperty(a)&&delete this.h[a];this.Wa=this.config=this.api=null;delete this.o;delete this.i;J.prototype.X.call(this)};
function qE(a){var b,c;a=(b=a.config)==null?void 0:(c=b.args)==null?void 0:c.fflags;return!!a&&a.indexOf("player_destroy_old_version=true")!==-1}
function lE(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.jsUrl:(a=a.config.assets)?a.js:""}
function pE(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.cssUrl:(a=a.config.assets)?a.css:""}
function oE(a,b){if(a.webPlayerContextConfig)var c=a.webPlayerContextConfig.serializedExperimentFlags;else{var d;if((d=a.config)==null?0:d.args)c=a.config.args.fflags}return(c||"").split("&").includes(b+"=true")}
function hE(a){for(var b={},c=y(Object.keys(a)),d=c.next();!d.done;d=c.next()){d=d.value;var e=a[d];b[d]=typeof e==="object"?Mi(e):e}return b}
;var uE={},vE="player_uid_"+(Math.random()*1E9>>>0);function wE(a,b){var c="player",d=!1;d=d===void 0?!0:d;c=typeof c==="string"?Si(c):c;var e=vE+"_"+Qa(c),f=uE[e];if(f&&d)return xE(a,b)?f.api.loadVideoByPlayerVars(a.args||null):f.loadNewVideoConfig(a),f.api;f=new cE(c,e,a,b,void 0);uE[e]=f;f.addOnDisposeCallback(function(){delete uE[f.getId()]});
return f.api}
function xE(a,b){return b&&b.serializedExperimentFlags?b.serializedExperimentFlags.includes("web_player_remove_playerproxy=true"):a&&a.args&&a.args.fflags?a.args.fflags.includes("web_player_remove_playerproxy=true"):!1}
;var yE=new Map;function zE(a,b,c,d,e){b=new AE(a,b,c,d===void 0?function(){}:d,e===void 0?null:e);
yE.set(a,b)}
function AE(a,b,c,d,e){J.call(this);this.container=a;this.webPlayerContextConfig=b;this.h=c;this.Pc=d;this.playerVars=e;BE(this)}
v(AE,J);function BE(a){if(G("yt.player.Application.create"))Promise.resolve().then(function(){CE(a)});
else{DE(Ep(a.webPlayerContextConfig.trustedJsUrl),function(){CE(a)},function(){a.I||a.Pc()});
var b=a.webPlayerContextConfig.trustedCssUrl;b&&EE(Ep(b))}}
function CE(a){if(!a.I){var b=G("yt.player.Application.create");try{a.api=b(a.container,{args:a.playerVars},a.webPlayerContextConfig,void 0).getInternalApi();var c=a.api.addEventListener;a.api.addEventListener=function(d,e){d==="onReady"?Promise.resolve().then(function(){e(a.api)}):c(d,e)};
a.api.isReady=function(){return!0};
a.h(a.api)}catch(d){throw a.Pc(),d;}}}
AE.prototype.X=function(){this.api&&this.api.destroy();Ui(this.container);J.prototype.X.call(this)};
function EE(a){var b="ytp-"+a.toString();if(!document.getElementById(b)){var c=document.createElement("link");c.id=b;Rb(c,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(c)}}
function DE(a,b,c){var d="ytp-"+a.toString(),e=document.getElementById(d);if(e)e.dataset.failed?c():e.dataset.loaded?b():(e.addEventListener("error",function(){c()}),e.addEventListener("load",function(){b()}));
else{var f=document.createElement("script");f.id=d;f.addEventListener("error",function(){f.dataset.failed="true";c()});
f.addEventListener("load",function(){f.dataset.loaded="true";b()});
Lb(f,a);a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(f,a.firstChild)}}
;var FE=null,GE=null,HE;function IE(a){FE=a;FE.addEventListener("onVideoDataChange",JE);FE.addEventListener("onReady",KE);a=S("POST_MESSAGE_ID","player");var b=S("POST_MESSAGE_ORIGIN");S("ENABLE_JS_API")?GE=new LD(FE,HE):S("ENABLE_POST_API")&&typeof a==="string"&&typeof b==="string"&&(GE=new TD(FE,a,b));HE=void 0}
function LE(){DA();T("ytidb_create_logger_embed_killswitch")||Bs();var a={};sD.h||(sD.h=new sD);sD.h.install((a.flush_logs={callback:function(){Wx()}},a));
T("embeds_disable_ios_networkless_logging")&&Js||rw();ED();vn.ra(function(){UC()});
a=Iq("att_init_delay",200);T("enable_rta_manager")&&setTimeout(function(){T("attmusi")&&fD();var b=new UB;var c={preload:!T("enable_rta_npi"),ye:T("attmusi")},d=!1;if(typeof c==="boolean")var e={preload:c};else typeof c==="undefined"?e={preload:!0}:(e=c,d=!!c.Xi);c=d?void 0:new fw;HB.instance=new HB(b,e,c);b=HB.instance;if(T("attmusi")&&T("attmusi_ue")){b={s:b.i.bind(b),ir:b.j.bind(b)};e=window;e.attmp=b;if(e.attmq!==void 0)for(c=y(e.attmq),d=c.next();!d.done;d=c.next())d=d.value,d(b);e.attmq=void 0}else e=
b.i.bind(b),E("yt.aba.att",e),b=b.j.bind(b),E("yt.aba.att2",b)},a);
Or(function(){if(T("enable_zw_ping")){var b=S("INNERTUBE_CLIENT_NAME","UNKNOWN_INTERFACE"),c="/establish_zw";b==="WEB_EMBEDDED_PLAYER"?c="/embed/establish_zw":b==="TVHTML5"&&(c="https://www.youtube.com/tv/establish_zw");S("COOKIELESS",!1)&&b==="WEB_EMBEDDED_PLAYER"?(b=new Headers,b.set("X-Goog-Visitor-Id",S("VISITOR_DATA")),fetch(c,{method:"GET",mode:"no-cors",headers:b})):fetch(c,{method:"GET",mode:"no-cors",credentials:"include"})}})}
function ME(){FB();var a=Br(),b=Er(119),c=window.devicePixelRatio>1;if(document.body&&Dn(document.body,"exp-invert-logo"))if(c&&!Dn(document.body,"inverted-hdpi")){var d=document.body;if(d.classList)d.classList.add("inverted-hdpi");else if(!Dn(d,"inverted-hdpi")){var e=Bn(d);Cn(d,e+(e.length>0?" inverted-hdpi":"inverted-hdpi"))}}else!c&&Dn(document.body,"inverted-hdpi")&&En();if(b!=c){b="f"+(Math.floor(119/31)+1);d=Fr(b)||0;d=c?d|67108864:d&-67108865;d===0?delete yr[b]:(c=d.toString(16),yr[b]=c.toString());
c=!0;T("web_secure_pref_cookie_killswitch")&&(c=!1);b=a.h;d=[];for(f in yr)yr.hasOwnProperty(f)&&d.push(f+"="+encodeURIComponent(String(yr[f])));var f=d.join("&");ur(b,f,63072E3,a.i,c)}}
function JE(){NE()}
function KE(){zB("ep_init_pr");NE()}
function NE(){var a=FE.getVideoData(1);a=a.title?a.title+" - YouTube":"YouTube";document.title!==a&&(document.title=a)}
function OE(){FE&&FE.sendAbandonmentPing&&FE.sendAbandonmentPing();S("PL_ATT")&&$D.dispose();for(var a=vn,b=0,c=AA.length;b<c;b++)a.sa(AA[b]);AA.length=0;zz(BA.toString());CA=!1;gq("DCLKSTAT",0);zc(GE);FE&&(FE.removeEventListener("onVideoDataChange",JE),FE.destroy(),FE=null)}
;zB("ep_init_eps");E("yt.setConfig",gq);E("yt.config.set",gq);E("yt.setMsg",wz);E("yt.msgs.set",wz);E("yt.logging.errors.log",ry);
E("writeEmbed",function(){zB("ep_init_wes");var a=S("PLAYER_CONFIG");if(!a){var b=S("PLAYER_VARS");b&&(a={args:b})}nA(!0);a.args.ps==="gvn"&&(document.body.style.backgroundColor="transparent");a.attrs||(a.attrs={width:"100%",height:"100%",id:"video-player"});b=document.referrer;window!==window.top&&b&&b!==document.URL&&(a.args.loaderUrl=b);b=IC();if(!b.serializedForcedExperimentIds){var c=uq(window.location.href);c.forced_experiments&&(b.serializedForcedExperimentIds=c.forced_experiments)}var d;((d=
a.args)==null?0:d.autoplay)?uB("watch",["pbs","pbu","pbp"]):a.args&&Zy(a.args)?uB("video_preview",["ol"]):uB("embed_no_video",["ep_init_ar"]);T("embeds_use_player_instances_library")?zE(document.getElementById("player"),b,function(e){IE(e)},function(){throw Error("Unable to load player JS");
},a.args):(a=wE(a,b),IE(a));
LE();zB("ep_init_wee")});
E("yt.abuse.player.botguardInitialized",G("yt.abuse.player.botguardInitialized")||aE);E("yt.abuse.player.invokeBotguard",G("yt.abuse.player.invokeBotguard")||bE);E("yt.abuse.dclkstatus.checkDclkStatus",G("yt.abuse.dclkstatus.checkDclkStatus")||EA);E("yt.player.exports.navigate",G("yt.player.exports.navigate")||mA);E("yt.util.activity.init",G("yt.util.activity.init")||Fw);E("yt.util.activity.getTimeSinceActive",G("yt.util.activity.getTimeSinceActive")||Jw);
E("yt.util.activity.setTimestamp",G("yt.util.activity.setTimestamp")||Gw);window.addEventListener("load",kq(function(){ME()}));
window.addEventListener("pageshow",kq(function(a){a.persisted||ME()}));
window.addEventListener("pagehide",kq(function(a){T("embeds_web_enable_dispose_player_if_page_not_cached_killswitch")?OE():a.persisted||OE()}));
T("embeds_enable_contrib_error_handling")?HC():(window.onerror=function(a,b,c,d,e){uy(a,b,c,d,e)},Xk=ty,window.addEventListener("unhandledrejection",function(a){ty(a.reason)}),qy());
(function(){if(S("ENABLE_JS_API")){var a=function(b){HE=b;window.removeEventListener("message",a)};
window.addEventListener("message",a)}})();
zB("ep_init_epe");}).call(this);
